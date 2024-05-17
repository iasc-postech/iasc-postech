import numpy as np
from PIL import Image, ImageDraw
import argparse
import random
from tqdm import tqdm

from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt

import torch
import json
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import numpy

import panopticapi
from panopticapi.utils import id2rgb, rgb2id

import os
from copy import deepcopy
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from torchvision.utils import draw_bounding_boxes, save_image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

def IrregularBox(
    mask_size, #(w,h)
    bbox,
    min_num_vertex = 15,
    max_num_vertex = 30):
    
    W, H = mask_size
    mask = Image.new('L', (W, H), 0)
    x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cx, cy = int((x1+x2)/2) , int((y1+y2)/2)
    box_width = int(x2-x1)
    box_height = int(y2-y1)
    num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
    
    vertex = []
    for i in range(num_vertex):
        vertex.append((int(cx+np.random.randint(-(box_width//2), (box_width//2))), int(cy+np.random.randint(-(box_height//2), (box_height//2)))))

    draw = ImageDraw.Draw(mask)
    width = int(box_width * 0.5)
    # draw.rounded_rectangle(xy=[x1,y1,x2,y2], radius=10, fill=1)
    draw.rectangle(xy=[x1,y1,x2,y2], fill=1)
    draw.line(vertex, fill=1, width=width)
    
    for v in vertex:
        draw.ellipse((v[0] - width//2,
                      v[1] - width//2,
                      v[0] + width//2,
                      v[1] + width//2),
                     fill=1)
        
    mask = np.asarray(mask, np.uint8)
    
    return mask

def MultipleIrregularBox(
    mask_size,
    bboxes,
    min_num_vertex = 15,
    max_num_vertex = 30):

    W, H = mask_size
    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)

    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        cx, cy = int((x1+x2)/2) , int((y1+y2)/2)
        box_width = int(x2-x1)
        box_height = int(y2-y1)
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        
        vertex = []
        for i in range(num_vertex):
            vertex.append((int(cx+np.random.randint(-(box_width//2), (box_width//2))), int(cy+np.random.randint(-(box_height//2), (box_height//2)))))

        width = int(box_width * 0.5)
        # draw.rounded_rectangle(xy=[x1,y1,x2,y2], radius=10, fill=1)
        draw.rectangle(xy=[x1,y1,x2,y2], fill=1)
        draw.line(vertex, fill=1, width=width)
        
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                        v[1] - width//2,
                        v[0] + width//2,
                        v[1] + width//2),
                        fill=1)
        
    mask = np.asarray(mask, np.uint8)
    
    return mask

def MakeMaskSingleBox(mask_size, bbox):
    mask = 1 - IrregularBox(mask_size, bbox) ## zero for the missing region, one for the remaining region
    return mask[np.newaxis, ...].astype(np.float32) 

def MakeMaskMultipleBox(mask_size, bboxes):
    mask = 1 - MultipleIrregularBox(mask_size, bboxes) ## zero for the missing region, one for the remaining region
    return mask[np.newaxis, ...].astype(np.float32) 

def main(args):
    torch.set_grad_enabled(False)
    # These are the COCO classes
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(512),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
    model.eval().to('cuda:0')

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
    blip2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16)
    blip2.to('cuda:0')

    # MAKE DATASET
    for fold in ['train2017', 'val2017']:
        with open(f'{args.coco_path}/annotations/panoptic_{fold}.json', 'r') as file:
            COCO_panop_data = json.load(file)

        image_base_path = f"{args.coco_path}/{fold}"
        save_base_path = f"{args.modified_coco_path}/{fold}"

        score_obj_quries_bboxes_path = os.path.join(save_base_path, 'score_obj_quries_bboxes')
        input_image_path = os.path.join(save_base_path, 'input')
        resize_image_path = os.path.join(save_base_path, 'image512')
        resize_mask_path = os.path.join(save_base_path, 'mask512')
        predict_pseg_path = os.path.join(save_base_path, 'predict_panoptic_seg')
        json_path = os.path.join(save_base_path, 'modified_data.json')
        
        os.makedirs(score_obj_quries_bboxes_path, exist_ok=True)
        os.makedirs(input_image_path, exist_ok=True)
        os.makedirs(resize_image_path, exist_ok=True)
        os.makedirs(resize_mask_path, exist_ok=True)
        os.makedirs(predict_pseg_path, exist_ok=True)

        bbox_target_layer = model.detr.bbox_embed
        detect_class_target_layer = model.detr.class_embed
        obj_queries_target_layer = model.detr.transformer
        # obj_queries_target_layer = model.detr.transformer.decoder

        category_list = [11,13,16 

                    ,16,17,18,19,20,21,22,24,25

                    ,44,46,47,51

                    ,53,55,56,58,59,60,61

                    ,70,72,73,74,85,86]

        def hook_bbox(module, input, output):
            bbox_list.append(output)

        def hook_class(module, input, output):
            detect_class_list.append(output)

        def hook_queries(module, input, output):
            obj_queries_list.append(output)

        obj_queries_target_layer.register_forward_hook(hook_queries)


        res = 512
        json_data_list = []
        for ann_per_image in tqdm(COCO_panop_data['annotations']):

            file_name = ann_per_image['file_name']
            image_path = os.path.join(image_base_path, file_name.replace('.png', '.jpg'))
            im = Image.open(image_path).convert('RGB')
            ori_w, ori_h = im.size 

            # CENTER CROP ORIGINAL IMAGE
            crop_width = crop_height = crop_res = min(ori_w, ori_h)
            left = (ori_w - crop_res) // 2
            top = (ori_h - crop_res) // 2
            right = left + crop_res
            bottom = top + crop_res
            center_cropped_image = im.crop((left, top, right, bottom))

            # RESIZE ORIGINAL IMAGE
            center_cropped_image = center_cropped_image.resize((res,res), resample=Image.BICUBIC)
            scale = res / crop_res

            ## Renew box coordinate according to the crop operation
            for segment in ann_per_image['segments_info']:

                box_x, box_y, box_w, box_h = segment['bbox']
                # CENTER CROP FIRST 
                box_x = box_x - left
                box_y = box_y - top
                if not (box_x > 0 and box_x + box_w < crop_res):
                    segment['bbox'] = [-1, -1, -1, -1]
                    continue
                if not (box_y > 0 and box_y + box_h < crop_res):
                    segment['bbox'] = [-1, -1, -1, -1]
                    continue
                
                # THEN REIZE THE BBOX
                box_x = box_x * scale
                box_y = box_y * scale
                box_w = box_w * scale
                box_h = box_h * scale

                segment['bbox'] = [box_x, box_y, box_w, box_h]


            for index, segment in enumerate(ann_per_image['segments_info']):

                ## select which class of instance will be removed
                if segment['category_id'] not in category_list:
                    continue
                if -1 in segment['bbox']:
                    continue
                box_x, box_y, box_w, box_h = segment['bbox']

                ## relative box coord
                box_x_rel, box_y_rel = box_x/res, box_y/res
                box_w_rel, box_h_rel = box_w/res, box_h/res

                ## if box condition does not satisfited, we skip this instance. 
                if box_w_rel < 0.1 or box_w_rel > 0.7:
                    continue        
                if box_h_rel < 0.1 or box_h_rel > 0.7:
                    continue


                resized_box_512 = [int(box_x_rel*res), int(box_y_rel*res), int(box_x_rel*res+box_w_rel*res), int(box_y_rel*res+box_h_rel*res)]
                mask_array = MakeMaskSingleBox((res, res) , resized_box_512).squeeze()
                pil_mask_image = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
                pil_mask_image.save(os.path.join(resize_mask_path, file_name.split('.')[0]+f'_{index}.png'))
                center_cropped_image.save(os.path.join(resize_image_path, file_name.split('.')[0]+f'_{index}.png'))

                mask = torch.tensor(mask_array[np.newaxis,:,:]).bool()
                image_mask = mask.repeat(3,1,1)

                img = transform(center_cropped_image)
                white_img = torch.ones_like(img)
                masked_out_img = torch.where(image_mask, img, white_img)

                obj_queries_list = []
                out = model(masked_out_img.unsqueeze(0).to('cuda:0'))
                masked_out_img = torch.round((masked_out_img*127.5 + 128).clamp(0.0, 255.0)).to(torch.uint8)
                save_image(masked_out_img/255, os.path.join(input_image_path, file_name.split('.')[0]+f'_{index}.png'))

                scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0].squeeze(0)
                bboxes = out["pred_boxes"].squeeze()
                obj_queries = obj_queries_list[0][0][-1].squeeze()
                numpy.savez(os.path.join(score_obj_quries_bboxes_path, file_name.split('.')[0]+f'_{index}.npz'), score=scores.cpu(), obj_queries=obj_queries.cpu(), pred_boxes=bboxes.cpu())

                ## the post-processor expects as input the target size of the predictions (which we set here to the image size)
                result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]
                panoptic_seg = Image.open(io.BytesIO(result['png_string']))
                panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
                panoptic_seg_id = rgb2id(panoptic_seg)
                panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

                ## We extract the segments info and the panoptic result from DETR's prediction
                segments_info = deepcopy(result["segments_info"])
                numpy.savez(os.path.join(predict_pseg_path, file_name.split('.')[0]+f'_{index}.npz'), seg=panoptic_seg_id, seg_info=segments_info)

                ## BLIP2 Captioning
                bigger_resized_box_512 = [max(0, resized_box_512[0]-box_w_rel*res*0.05), max(0, resized_box_512[1]-box_h_rel*res*0.05), min(res, resized_box_512[2]+box_w_rel*res*0.05), min(res, resized_box_512[3]+box_h_rel*res*0.05)]
                # bbox_area = center_cropped_image.crop(resized_box_512).convert('RGB')
                bigger_bbox_area = center_cropped_image.crop(bigger_resized_box_512).convert('RGB')
                inputs = processor(bigger_bbox_area, return_tensors="pt").to("cuda:0", torch.float16)
                generated_ids = blip2.generate(**inputs)
                generated_text = [i.strip() for i in processor.batch_decode(generated_ids, skip_special_tokens=True)][0]

                json_data = {}
                json_data['missing_class'] = segment['category_id']
                json_data['missing_bbox'] = [box_x_rel, box_y_rel, box_x_rel+box_w_rel, box_y_rel+box_h_rel]
                json_data['missing_instance_id'] = segment['id']
                json_data['score_obj_quries_bboxes'] = file_name.split('.')[0]+f'_{index}'
                json_data['blip2_caption'] = generated_text
                json_data_list.append(json_data)

        with open(json_path, "w") as json_file:
            json.dump(json_data_list, json_file, indent=4)

    print("finished generating dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", type=str, default="./coco", help="path to coco dataset")
    parser.add_argument("--modified_coco_path", type=str, default="./modified_coco", help="path to save modified coco dataset")
    args = parser.parse_args()
    main(args)