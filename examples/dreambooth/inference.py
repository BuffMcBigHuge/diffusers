# inference.py

try:
    import torch
    from torch import autocast
    from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
    import sys
    import argparse
    import secrets

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
            type=str,
            default="",
            help="guidance scale,
        )

        opt = parser.parse_args()

        # RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='${HOME}/default-models/RealESRGAN_x2plus/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True)  # need to set False in CPU mode

        # GFPGANer
        restorer = GFPGANer(
            model_path='${HOME}/default-models/GFPGANv1.4/GFPGANv1.4.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_upsampler)

        # STABLE DIFFUSION
        scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = StableDiffusionPipeline.from_pretrained(opt.modeldir, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")

        g_cuda = None
        g_cuda = torch.Generator(device='cuda')
        seed = opt.seed #@param {type:"number"}
        g_cuda.seed()
        
        prompt = opt.prompt #@param {type:"string"}
        negative_prompt = opt.negative_prompt #@param {type:"string"}
        n_samples = opt.n_samples #@param {type:"number"}
        num_inference_steps = opt.ddim_steps #@param {type:"number"}
        height = opt.h #@param {type:"number"}
        width = opt.w #@param {type:"number"}
        guidance_scale = opt.cfg #@param {type:"number"}

        with autocast("cuda"), torch.inference_mode():
            images = pipe(
                prompt,
                height,
                width,
                negative_prompt,,
                num_images_per_prompt=n_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=g_cuda
            ).images

        for image in images:
        
            # read image
            # img_name = os.path.basename(img_path)
            # basename, ext = os.path.splitext(img_name)
            # input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # restore faces and background if necessary
            restored_img = restorer.enhance(
                input_img,
                has_aligned='auto',
                only_center_face=True,
                paste_back=True,
                weight=0.5)

            # save image to disk
            restored_img.save(opt.outdir + '/' + secrets.token_urlsafe(16) + '.jpg', 'jpeg', quality=100)
        exit(0)

    if __name__ == "__main__":
        main()
except:
  print(f"ERROR")
  exit(1)