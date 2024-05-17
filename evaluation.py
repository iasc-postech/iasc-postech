import torch
from PIL import Image
import json
import os
from tqdm import tqdm
import argparse

from iasc_utils.metric import VisualGroundingAcc, CLIPscore  
from torchmetrics.image.fid import FrechetInceptionDistance

from kornia.geometry.transform import crop_and_resize
import torchvision.transforms as transforms
import torchvision.transforms.functional


def main(args):
    annotation_path = args.annotation_path
    repaint_bpath = args.repaint_bpath
    gt_bpath = args.gt_bpath

    with open(annotation_path, "r") as file:
        file_list = json.load(file)

    fid = FrechetInceptionDistance(feature=2048).to('cuda')
    CLIPScore = CLIPscore().to('cuda')
    vga = VisualGroundingAcc().to('cuda')
    count = 0

    with torch.no_grad():
        for file_ann in tqdm(file_list):
            file_detail_name = file_ann['score_obj_quries_bboxes']
            clip_caption = 'A photo of ' + file_ann['blip2_caption']
            bbox = file_ann['missing_bbox']

            # visual_grounding_
            transform_list = []
            transform_list += [transforms.ToTensor()]
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5))]
            transform = transforms.Compose(transform_list)

            repaint_path = os.path.join(repaint_bpath, file_detail_name +'.png')
            gt_path = os.path.join(gt_bpath, file_detail_name +'.png')
        
        
            if not os.path.exists(gt_path):
                continue
            if not os.path.exists(repaint_path):
                count += 1
                continue
            
            repaint = Image.open(repaint_path).convert(mode='RGB').resize((512,512),Image.Resampling.BICUBIC)
            gt = Image.open(gt_path).convert(mode='RGB').resize((512,512),Image.Resampling.BICUBIC)
            repaint = transform(repaint).unsqueeze(0).to('cuda')
            gt = transform(gt).unsqueeze(0).to('cuda')

            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            crop_y1, crop_y2, crop_x1, crop_x2 = int(y1*512), int(y2*512), int(x1*512), int(x2*512)
            crop_region = torch.tensor([[crop_x1, crop_y1],[crop_x2, crop_y1],[crop_x2, crop_y2],[crop_x1, crop_y2]]).unsqueeze(0)
            repaint_crop = crop_and_resize(repaint, crop_region, size=(512, 512), mode='bicubic')

            fid.update(imgs=((gt+1)*0.5).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8), real=True)
            fid.update(imgs=((repaint+1)*0.5).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8), real=False)

            CLIPScore.update(repaint_crop, type='Image')
            CLIPScore.update(clip_caption, type='Text')

            vg_target = {
                'path': [file_detail_name],
                'ori_w' : torch.tensor([512]).to('cuda'),
                'ori_h' : torch.tensor([512]).to('cuda'),
                'region_coords' : torch.tensor([[crop_x1, crop_y1, crop_x2, crop_y2]]).to('cuda'),
                'patch_mask' : torch.tensor([[True]]).to('cuda'),
                'blip2_caption' : [file_ann['blip2_caption']]
            }
            vga.update(repaint, vg_target)

    # print(count)
    print("FID: ", fid.compute().item())
    print("CLIPScore: ",  CLIPScore.compute()[0])
    print("Visual Grounding Acc: ", vga.compute().item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, required=True, help="path to annotation file")
    parser.add_argument("--repaint_bpath", type=str, required=True, help="path to repaint image")
    parser.add_argument("--gt_bpath", type=str, required=True, help="path to gt image")
    args = parser.parse_args()
    main(args)