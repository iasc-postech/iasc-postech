import argparse
import torch
import numpy as np
import PIL.Image as Image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm
import os

def main(args):
    fid = FrechetInceptionDistance(feature=2048).to('cuda')
    with torch.no_grad():
        for gt_image in tqdm(os.listdir(args.gt_folder), dynamic_ncols=True):
            gt_image = Image.open(os.path.join(args.gt_folder, gt_image)).convert('RGB')
            gt_image = np.array(gt_image)
            gt_image = torch.tensor(gt_image).permute(2, 0, 1).unsqueeze(0).to('cuda').to(torch.uint8)
            fid.update(imgs=gt_image, real=True)
        for pred_image in tqdm(os.listdir(args.pred_folder), dynamic_ncols=True):
            pred_image = Image.open(os.path.join(args.pred_folder, pred_image)).convert('RGB')
            pred_image = np.array(pred_image)
            pred_image = torch.tensor(pred_image).permute(2, 0, 1).unsqueeze(0).to('cuda').to(torch.uint8)
            fid.update(imgs=pred_image, real=False)
    
    print("FID Score: ", fid.compute().item())           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_folder", type=str, required=True, help="path to input folder 1")
    parser.add_argument("--pred_folder", type=str, required=True, help="path to input folder 2")
    args = parser.parse_args()
    main(args)