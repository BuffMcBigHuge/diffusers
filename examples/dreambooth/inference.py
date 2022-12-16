# inference.py

import time
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler

import cv2
import os
import sys
import argparse
import secrets
import math

from PIL import Image
import numpy as np
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="",
        help="negative prompt"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="how many samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.5,
        help="guidance scale",
    )

    opt = parser.parse_args()

    # RealESRGANer
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='/home/ubuntu/default-models/RealESRGAN_x2plus/RealESRGAN_x2plus.pth',
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True)  # need to set False in CPU mode

    # GFPGANer
    restorer = GFPGANer(
        model_path='/home/ubuntu/default-models/GFPGANv1.4/GFPGANv1.4.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=bg_upsampler)

    # STABLE DIFFUSION
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    pipe = StableDiffusionPipeline.from_pretrained(opt.modeldir, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

    # LOAD IMG2IMG on SAME COMPONENTS (shared VRAM)
    components = pipe.components
    imgpipe = StableDiffusionImg2ImgPipeline(**components)
    imgpipe.enable_xformers_memory_efficient_attention()
    imgpipe.to("cuda")

    g_cuda = None
    g_cuda = torch.Generator(device='cuda')
    seed = opt.seed #@param {type:"number"}
    g_cuda.seed()
    
    prompt = opt.prompt #@param {type:"string"}
    negative_prompt = opt.negative_prompt #@param {type:"string"}
    num_images_per_prompt = opt.samples #@param {type:"number"}
    num_inference_steps = opt.ddim_steps #@param {type:"number"}
    height = opt.height #@param {type:"number"}
    width = opt.width #@param {type:"number"}
    guidance_scale = opt.cfg #@param {type:"number"}

    ## HIGHRES FIX
    desired_pixel_count = 512 * 512
    actual_pixel_count = width * height
    scale = math.sqrt(desired_pixel_count / actual_pixel_count)
    firstphase_width = math.ceil(scale * width / 64) * 64
    firstphase_height = math.ceil(scale * height / 64) * 64

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=firstphase_height,
            width=firstphase_width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images        

    for image in images:        
        # resize
        image = image.resize((width, height)) # resizeimage.resize_cover(image, [width, height])
        image = image.convert("RGB")
        
         # img2img
        with autocast("cuda"), torch.inference_mode():
            image = imgpipe(
                prompt=prompt,
                init_image=image,
                strength=0.75,
                negative_prompt=negative_prompt,
                num_images_per_prompt=1,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=g_cuda
            ).images[0]
        
        # restore faces and background if necessary
        input_image = np.array(image, dtype=np.uint8)
        np_image_bgr = input_image[:, :, ::-1]
        cropped_faces, restored_faces, gfpgan_output_bgr = restorer.enhance(
            np_image_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5)
        np_image = gfpgan_output_bgr[:, :, ::-1]
        output_image = Image.fromarray(np_image)
        
        # save image to disk
        reverse_timestamp = 10000000000 - int(time.time())
        save_image_path = os.path.join(opt.outdir, f'{reverse_timestamp}-{secrets.token_hex(16)}.jpg')
        output_image.save(save_image_path, 'jpeg', quality=80)
    exit(0)

if __name__ == "__main__":
    main()