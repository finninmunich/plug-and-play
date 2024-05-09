import argparse
import json
import numpy as np
import os
import torch
from PIL import Image
from contextlib import nullcontext
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    # image = transforms.CenterCrop(min(x, y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="configs/pnp/pnp-all-sunnyday2snowynight-v1.yaml",
        help="path to the feature extraction config file"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--save_all_features",
        default=False,
        action="store_true",
        help="if set to true, saves all feature maps, otherwise only saves those necessary for PnP",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--check-safety",
        action='store_true',
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=9999999,
        help="number of images to prcess",
    )
    opt = parser.parse_args()
    setup_config = OmegaConf.load("./configs/pnp/setup.yaml")
    model_config = OmegaConf.load(f"{opt.model_config}")
    exp_config = OmegaConf.load(f"{opt.config}")
    exp_path_root = setup_config.config.exp_path_root
    seed = 1
    seed_everything(seed)  # seed everything

    model = load_model_from_config(model_config, exp_config.config.ckpt)  # load model

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    save_feature_timesteps = exp_config.config.save_feature_timesteps
    sampler = DDIMSampler(model)
    outpath = os.path.join(exp_path_root,exp_config.config.experiment_name+'_inversion_prompts')

    os.makedirs(outpath, exist_ok=True)

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        # save_sampled_img(pred_x0, i, predicted_samples_path)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block in blocks:
            if not opt.save_all_features and block_idx < 4:
                block_idx += 1
                continue
            if "ResBlock" in str(type(block[0])):
                if opt.save_all_features or block_idx == 4:
                    feature_map_dict[f"{feature_type}_{block_idx}_in_layers_features_time_{i}"] = block[
                        0].in_layers_features
                    feature_map_dict[f"{feature_type}_{block_idx}_out_layers_features_time_{i}"] = block[
                        0].out_layers_features
                    # save_feature_map(block[0].in_layers_features,
                    #                  f"{feature_type}_{block_idx}_in_layers_features_time_{i}")
                    # save_feature_map(block[0].out_layers_features,
                    #                  f"{feature_type}_{block_idx}_out_layers_features_time_{i}")
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                feature_map_dict[f"{feature_type}_{block_idx}_self_attn_k_time_{i}"] = \
                    block[1].transformer_blocks[0].attn1.k
                feature_map_dict[f"{feature_type}_{block_idx}_self_attn_q_time_{i}"] = \
                    block[1].transformer_blocks[0].attn1.q
                # save_feature_map(block[1].transformer_blocks[0].attn1.k,
                #                  f"{feature_type}_{block_idx}_self_attn_k_time_{i}")
                # save_feature_map(block[1].transformer_blocks[0].attn1.q,
                #                  f"{feature_type}_{block_idx}_self_attn_q_time_{i}")
            block_idx += 1

    def save_feature_maps_callback(i):
        if opt.save_all_features:
            save_feature_maps(unet_model.input_blocks, i, "input_block")
        save_feature_maps(unet_model.output_blocks, i, "output_block")


    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                # img_file_list = os.listdir(opt.input_folder)
                # img_file_list = [f for f in img_file_list if not f.endswith(".jsonl")]
                json_file_path = os.path.join(exp_config.config.input_folder, "metadata.jsonl")
                img_file_list = []
                inversion_prompts_list=[]
                prompts_list = []
                with open(json_file_path, 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        txt = data['text']
                        if exp_config.config.source_weather in txt:
                            img_file_list.append(data['file_name'])
                            inversion_prompts_list.append(data['text'])
                            prompts_list.append(data['text'].replace(exp_config.config.source_weather,
                                                                     exp_config.config.target_weather))
                print(f"we totally have {len(img_file_list)} images in the folder.")
                if opt.num_images < len(img_file_list):
                    img_file_list = img_file_list[:opt.num_images]
                    print(f"we only process {opt.num_images} images.")
                for image, inversion_prompts,forward_prompt in zip(img_file_list, inversion_prompts_list,prompts_list):
                    # print(f"processing image {image}")
                    # print(f"inversion prompts: {inversion_prompts}")
                    # print(f"forward prompts: {forward_prompt}")
                    # print(f"feature extraction......")
                    image_file_path = os.path.join(exp_config.config.input_folder, image)
                    assert os.path.isfile(image_file_path)
                    init_image = load_img(image_file_path).to(device)
                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
                    # init_latent == z_0
                    ddim_inversion_steps = exp_config.config.ddim_inversion_steps
                    feature_map_dict = {}

                    # set condition/uncondition for ddim inversion

                    uc = model.get_learned_conditioning([""])
                    if isinstance(inversion_prompts, tuple):
                        inversion_prompts = list(inversion_prompts)
                    if not isinstance(inversion_prompts, list):
                        if isinstance(inversion_prompts, str):
                            inversion_prompts = [inversion_prompts]
                        else:
                            inversion_prompts = list(inversion_prompts)
                    c = model.get_learned_conditioning(inversion_prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                    # ddim inversion
                    z_enc, _ = sampler.encode_ddim(init_latent, num_steps=ddim_inversion_steps, conditioning=c,
                                                   unconditional_conditioning=uc,
                                                   unconditional_guidance_scale=1)
                    _, _ = sampler.sample(S=exp_config.config.ddim_steps,  # 100
                                          conditioning=c,
                                          batch_size=1,
                                          shape=shape,
                                          verbose=False,
                                          unconditional_guidance_scale=1,
                                          unconditional_conditioning=uc,
                                          eta=opt.ddim_eta,
                                          x_T=z_enc,
                                          img_callback=ddim_sampler_callback,
                                          callback_ddim_timesteps=save_feature_timesteps,
                                          outpath=outpath)

                    print(f"plug-and-play forward process......")
                    ddim_steps = exp_config.config.num_ddim_sampling_steps
                    foward_sampler = DDIMSampler(model)
                    foward_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
                    negative_prompt = exp_config.config.negative_prompt

                    def load_target_features():
                        self_attn_output_block_indices = [4, 5, 6, 7, 8, 9, 10, 11]
                        out_layers_output_block_indices = [4]
                        output_block_self_attn_map_injection_thresholds = [ddim_steps // 2] * len(
                            self_attn_output_block_indices)
                        feature_injection_thresholds = [exp_config.config.feature_injection_threshold]
                        target_features = []

                        time_range = np.flip(foward_sampler.ddim_timesteps)
                        total_steps = foward_sampler.ddim_timesteps.shape[0]
                        # print(f"time range: {time_range}")
                        # print(f"total steps: {total_steps}")
                        # assert False
                        iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)

                        for i, t in enumerate(iterator):
                            current_features = {}
                            for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(
                                    self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
                                if i <= int(output_block_self_attn_map_injection_threshold):
                                    output_q = feature_map_dict[f"output_block_{output_block_idx}_self_attn_q_time_{t}"]
                                    output_k = feature_map_dict[f"output_block_{output_block_idx}_self_attn_k_time_{t}"]
                                    # output_q = torch.load(os.path.join(source_experiment_qkv_path,
                                    #                                    f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt"))
                                    # output_k = torch.load(os.path.join(source_experiment_qkv_path,
                                    #                                    f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
                                    current_features[f'output_block_{output_block_idx}_self_attn_q'] = output_q
                                    current_features[f'output_block_{output_block_idx}_self_attn_k'] = output_k

                            for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices,
                                                                                       feature_injection_thresholds):
                                if i <= int(feature_injection_threshold):
                                    output = feature_map_dict[
                                        f"output_block_{output_block_idx}_out_layers_features_time_{t}"]
                                    # output = torch.load(os.path.join(source_experiment_out_layers_path,
                                    #                                  f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))
                                    current_features[f'output_block_{output_block_idx}_out_layers'] = output

                            target_features.append(current_features)

                        return target_features

                    batch_size = 1
                    start_code = z_enc
                    if start_code is not None:
                        start_code = start_code.repeat(batch_size, 1, 1, 1)

                    injected_features = load_target_features()
                    unconditional_prompt = ""
                    uc = None
                    nc = None
                    if exp_config.config.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [unconditional_prompt])
                        nc = model.get_learned_conditioning(batch_size * [negative_prompt])
                    assert forward_prompt is not None

                    if not isinstance(forward_prompt, list):
                        if isinstance(forward_prompt, str):
                            forward_prompt = [forward_prompt]
                        else:
                            forward_prompt = list(forward_prompt)
                    c = model.get_learned_conditioning(forward_prompt)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = foward_sampler.sample(S=ddim_steps,
                                                            conditioning=c,
                                                            negative_conditioning=nc,
                                                            batch_size=len(forward_prompt),
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=exp_config.config.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            img_callback=None,
                                                            injected_features=injected_features,
                                                            negative_prompt_alpha=exp_config.config.negative_prompt_alpha,
                                                            negative_prompt_schedule=exp_config.config.negative_prompt_schedule,
                                                            )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                    sample_idx = 0
                    for k, x_sample in enumerate(x_image_torch):
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(outpath, image))
                        sample_idx += 1


if __name__ == "__main__":
    main()
