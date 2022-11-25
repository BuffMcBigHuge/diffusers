# inference.py

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import sys
import argparse
import secrets

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
        "--n_samples",
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
        "--modeldir",
        type=str,
        default="",
        help="path to checkpoint of model",
    )

    opt = parser.parse_args()

    print("prompt: ", opt.prompt)

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    pipe = StableDiffusionPipeline.from_pretrained(opt.modeldir, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")

    g_cuda = None
    g_cuda = torch.Generator(device='cuda')
    seed = opt.seed #@param {type:"number"}
    g_cuda.seed()

    prompt = opt.prompt #@param {type:"string"}
    negative_prompt = opt.negative_prompt #@param {type:"string"}
    n_samples = opt.n_samples #@param {type:"number"}
    num_inference_steps = opt.ddim_steps #@param {type:"number"}
    height = opt.H #@param {type:"number"}
    width = opt.W #@param {type:"number"}
    guidance_scale = 7.5 #@param {type:"number"}

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=n_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    for image in images:
        # save image to disk
        image.save(opt.outdir + '/' + secrets.token_urlsafe(16) + '.jpg', 'jpeg', quality=100)

if __name__ == "__main__":
    main()