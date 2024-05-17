import numpy as np
from PIL import Image
import argparse
import os
import json
import utils
import torch
from diffusers import StableDiffusionInpaintPipeline
from tqdm.auto import tqdm

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    sd_pipe.run_safety_checker = utils.disable_safety_checker
    sd_pipe.set_progress_bar_config(disable=True)

    masked_image_path = os.path.join(args.root_folder, "masked_image")
    mask_path = os.path.join(args.root_folder, "mask")
    json_path = os.path.join(args.root_folder, "json_samples")
    
    files = sorted([i.split(".jpg")[0] for i in os.listdir(masked_image_path)])

    for file in tqdm(files, total=len(files), dynamic_ncols=True):
        img = Image.open(os.path.join(masked_image_path, file + ".jpg")).convert("RGB")
        img_mask = Image.open(os.path.join(mask_path, file + ".png")).convert("L")
        with open(os.path.join(json_path, file + ".json"), "r") as f:
            scenegraph_input = json.load(f)
        
        original_size = img.size
        utils.get_caption(scenegraph_input)
        if args.refine_type == 1:
            caption = utils.refine_caption1(scenegraph_input, img_mask)
        else:
            caption = utils.refine_caption2(scenegraph_input, img_mask)

        img, img_mask = utils.resize_for_stable_diffusion(img), utils.resize_for_stable_diffusion(img_mask, use_nearest=True)
        img_mask = utils.binarize_mask(img_mask)
        
        with torch.inference_mode():
            imgout = sd_pipe(
                prompt=caption, image=img, mask_image=img_mask, num_inference_steps=50, guidance_scale=7.5, height=img.height, width=img.width
            ).images[0]
            torch.cuda.empty_cache()

        if imgout.size != original_size:
            imgout = imgout.resize(original_size, Image.LANCZOS)

        imgout.save(os.path.join(args.output_folder, file + ".jpg"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help="Root folder containing the images"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output folder"
    )
    parser.add_argument(
        "--refine_type",
        type=int,
        default=2,
        help="Output folder"
    )
    args = parser.parse_args()
    main(args)