# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import os
from time import perf_counter

import click
import imageio

import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image

import clip
import dnnlib
import legacy

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

def bit_conversion_16_to_8(images: torch.Tensor):
    '''
    Convert 16-bit input images into 8-bit and return the converted images.
    '''
    converted = images.to(torch.float32) / 256
    converted = converted.clamp(0, 255)
    return converted

#----------------------------------------------------------------------------

def spherical_dist_loss(x: torch.Tensor, y: torch.Tensor):
    '''
    Original code by Katherine Crowson, copied from https://github.com/afiaka87/clip-guided-diffusion/blob/main/cgd/losses.py
    '''
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

#----------------------------------------------------------------------------

def project(
    G,
    target_image: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    target_text,
    *,
    num_steps                  = 512,
    w_avg_samples              = 8192,
    initial_learning_rate      = 0.1,
    initial_latent             = None,
    initial_noise_factor       = 0.01,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.5,
    noise_ramp_length          = 0.5,
    max_noise                  = 0.5,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    use_w_only                 = False,
    use_cosine_dist            = True,
    use_spherical_dist         = False,
    is_16_bit                  = False,
    device: torch.device
):
    if target_image is not None:
        assert target_image.shape[1:] == (G.img_resolution, G.img_resolution)
    
    assert use_cosine_dist or use_spherical_dist

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.randn(w_avg_samples, G.z_dim)
    labels = None
    if (G.mapping.c_dim):
        labels = torch.from_numpy(0.5*np.random.randn(w_avg_samples, G.mapping.c_dim)).to(device)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), labels)  # [N, L, C]
    w_samples = w_samples.cpu().numpy().astype(np.float32)                 # [N, L, C]
    w_samples_1d = w_samples[:, :1, :].astype(np.float32)

    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, L, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    std_dev = np.std(w_samples)

    if initial_latent is not None:
        w_avg = initial_latent
        if w_avg.shape[1] == 1 and not use_w_only:
            w_avg = np.tile(w_avg, (1, G.mapping.num_ws, 1))

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load CLIP
    model, transform = clip.load("ViT-B/16", device=device)

    # Features for target image.
    if target_image is not None:
        target_images = target_image.unsqueeze(0).to(device).to(torch.float32)
        if target_images.shape[2] > 224:
            target_images = F.interpolate(target_images, size=(224, 224), mode='area')
            # target_images = F.interpolate(target_images, size=(256, 256), mode='area')
            # target_images = target_images[:, :, 16:240, 16:240] # 256 -> 224, center crop
        with torch.no_grad():
            clip_target_features = model.encode_image(((target_images / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]).float()

    if use_w_only:
        w_avg = np.mean(w_avg, axis=1, keepdims=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    # optimizer = madgrad.MADGRAD([w_opt] + list(noise_bufs.values()), lr=initial_learning_rate)
    # optimizer = SM3.SM3([w_opt] + list(noise_bufs.values()), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = max_noise * w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        if use_w_only:
            ws = ws.repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 224x224 if it's larger than that. CLIP was built for 224x224 images.
        if is_16_bit:
            synth_images = (synth_images * 32767.5 + 32767.5).clamp(0, 65535)
            synth_images = bit_conversion_16_to_8(synth_images)
        else:
            synth_images = (torch.clamp(synth_images, -1, 1) + 1) * (255/2)
        if synth_images.shape[1] == 1:
            synth_images = synth_images.repeat([1, 3, 1, 1])

        synth_images = F.interpolate(synth_images, size=(224, 224), mode='area')
        # synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
        # synth_images = synth_images[:, :, 16:240, 16:240] # 256 -> 224, center crop

        synth_images = ((synth_images / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]

        dist = 0
        cosine_sim_loss_image, cosine_sim_loss_text = 0, 0
        spherical_loss_image, spherical_loss_text = 0, 0

        # adj_center = 2.0

        if target_image is not None:
            # Cosine Similarity
            if use_cosine_dist:
                generated_encoded = model.encode_image(synth_images).float()
                cosine_sim = torch.cosine_similarity(clip_target_features, generated_encoded, dim=-1).mean()
                cosine_sim_loss_image = 1 - cosine_sim
                # cosine_sim_loss_image = -1 * cosine_sim
                dist += cosine_sim_loss_image

            # Spherical Distance
            if use_spherical_dist:
                generated_encoded = model.encode_image(synth_images).float()
                spherical_loss_image = spherical_dist_loss(generated_encoded.unsqueeze(0), clip_target_features.unsqueeze(0)).sum()
                dist += spherical_loss_image

            # # Original
            # clip_dist = (clip_target_features - model.encode_image(synth_images).float()).square().sum()
            # dist += F.relu(0.5 + adj_center * clip_dist - min_threshold)

        if target_text is not None:
            # Cosine Similarity
            if use_cosine_dist:
                cosine_sim = (model(synth_images, target_text)[0] / 100).sum()
                # cosine_sim = torch.cosine_similarity(model.encode_text(target_text).float(), model.encode_image(synth_images).float(), dim=-1).mean()
                cosine_sim_loss_text = 1 - cosine_sim
                # cosine_sim_loss_text = -1 * cosine_sim
                dist += cosine_sim_loss_text

            # Spherical Distance
            if use_spherical_dist:
                generated_encoded = model.encode_image(synth_images).float()
                txt_encoded = model.encode_text(target_text).float()
                spherical_loss_text = spherical_dist_loss(generated_encoded.unsqueeze(0), txt_encoded.unsqueeze(0)).sum()
                dist += spherical_loss_text

            # # Original
            # clip_text = 1 - model(clip_synth_image, target_text)[0].sum() / 100
            # dist += 2 * F.relu(adj_center * clip_text * clip_text - min_threshold / adj_center)

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.2f} dist {dist:<4.2f} c_loss_text {cosine_sim_loss_text:<4.2f} s_loss_text {spherical_loss_text:<4.2f} reg_loss {reg_loss * regularize_noise_weight:<4.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',       help='Network pickle filename', required=True)
@click.option('--target-image', 'target_fname', help='Target image file to project to', required=False, metavar='FILE', default=None)
@click.option('--target-text',                  help='Target text to project to', required=False, default=None)
@click.option('--initial-latent',               help='Initial latent', default=None)
@click.option('--lr',                           help='Learning rate', type=float, default=0.3, show_default=True)
@click.option('--num-steps',                    help='Number of optimization steps', type=int, default=300, show_default=True)
@click.option('--seed',                         help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',                   help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                       help='Where to save the output images', required=True, metavar='DIR')
@click.option('--use-cosine-dist',              help='Use cosine distance when calculating the loss', type=bool, default=True, show_default=True)
@click.option('--use-spherical-dist',           help='Use spherical distance when calculating the loss', type=bool, default=False, show_default=True)
@click.option('--16bit', 'is_16_bit',           help='Set to true if the network is trained to output 16-bit images', type=bool, default=False, show_default=True)
@click.option('--use-w-only',                   help='Project into w space instead of w+ space', type=bool, default=False, show_default=True)
def run_projection(
    network_pkl: str,
    target_fname: str,
    target_text: str,
    initial_latent: str,
    lr: float,
    num_steps: int,
    seed: int,
    save_video: bool,
    outdir: str,
    use_cosine_dist: bool,
    use_spherical_dist: bool,
    is_16_bit: bool,
    use_w_only: bool,
):
    """Project given image to the latent space of pretrained network pickle using CLIP.

    Examples:

    \b
    python clip_search.py --outdir=out --target-text='An image of an apple.' \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    # Set seed value
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print(f'Loading networks from {network_pkl}...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_image = None
    if target_fname:
        target_pil = PIL.Image.open(target_fname).convert('RGB').filter(PIL.ImageFilter.SHARPEN)

        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target_image = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

    if target_text:
        target_text = clip.tokenize(target_text).to(device)
        # target_text = torch.cat([clip.tokenize(target_text)]).to(device)

    if initial_latent is not None:
        initial_latent = np.load(initial_latent)
        initial_latent = initial_latent[initial_latent.files[0]]

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target_image=target_image,
        target_text=target_text,
        initial_latent=initial_latent,
        initial_learning_rate=lr,
        num_steps=num_steps,
        is_16_bit=is_16_bit,
        use_w_only=use_w_only,
        use_cosine_dist=use_cosine_dist,
        use_spherical_dist=use_spherical_dist,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Save final projected frame and W vector.
    os.makedirs(outdir, exist_ok=True)
    if target_fname:
        target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    if is_16_bit:
        synth_image = (synth_image.permute(0, 2, 3, 1) * 32767.5 + 32767.5).clamp(0, 65535).to(torch.int32)
        synth_image = synth_image[0].cpu().numpy().astype(np.uint16)
        mode = 'I;16'
    else:
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        mode = 'RGB'
    PIL.Image.fromarray(synth_image, mode).save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    # Render debug output: optional video and projected image and W vector.
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            if is_16_bit:
                synth_image = (synth_image.permute(0, 2, 3, 1) * 32767.5 + 32767.5).clamp(0, 65535)
                synth_image = bit_conversion_16_to_8(synth_image)
                synth_image = synth_image[0].cpu().numpy().astype(np.uint8)
                synth_image = synth_image.repeat(3, axis=-1)
            else:
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            if target_fname:
                video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
            else:
                video.append_data(synth_image)
        video.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------