import os, sys
import numpy as np
import json

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from panopticapi.utils import IdGenerator, rgb2id

import argparse

def main(args):

    # prepare paths
    coco_path = args.coco_path
    modified_coco_path = args.modified_coco_path
    out_path = args.out_path

    train_selected_images = list(set([i.split('_')[0] for i in os.listdir(os.path.join(modified_coco_path, 'train2017', 'image512'))]))
    val_selected_images = list(set([i.split('_')[0] for i in os.listdir(os.path.join(modified_coco_path, 'val2017', 'image512'))]))

    train_json_file = os.path.join(coco_path, 'annotations/panoptic_train2017.json')
    val_json_file = os.path.join(coco_path, 'annotations/panoptic_val2017.json')

    train_segmentations_folder = os.path.join(coco_path, 'annotations/panoptic_train2017')
    val_segmentations_folder = os.path.join(coco_path, 'annotations/panoptic_val2017')

    train_img_folder = os.path.join(coco_path, 'train2017')
    val_img_folder = os.path.join(coco_path, 'val2017')

    panoptic_coco_categories = './coco/annotations/panoptic_coco_categories.json'

    train_out_folder = os.path.join(out_path, 'train2017')
    val_out_folder = os.path.join(out_path, 'val2017')


    with open(train_json_file, 'r') as f:
        train_coco_d = json.load(f)

    with open(val_json_file, 'r') as f:
        val_coco_d = json.load(f)

    with open(panoptic_coco_categories, 'r') as f:
        categories_list = json.load(f)

    categegories = {category['id']: category for category in categories_list}

    for i in range(0, 91):
        os.makedirs(os.path.join(train_out_folder, str(i)), exist_ok=True)
        os.makedirs(os.path.join(val_out_folder, str(i)), exist_ok=True)


    # train instance mask generation
    for ann in train_coco_d['annotations']:
        if ann['file_name'].split('.')[0] not in train_selected_images:
            continue
        segmentation = np.array(
        Image.open(os.path.join(train_segmentations_folder, ann['file_name'])),
        dtype=np.uint8)
        segmentation_id = rgb2id(segmentation)
        for segment_info in ann['segments_info']:

            if segment_info['category_id'] < 91:
                
                instance_class = segment_info['category_id']
                x, y, width, height = segment_info['bbox']
                center_x, center_y = (x+(x+width))//2, (y+(y+height))//2
                box_length = max(width, height)

                if box_length < 32:
                    continue

                min_x, max_x = center_x - box_length//2, center_x + box_length//2
                min_y, max_y = center_y - box_length//2, center_y + box_length//2

                instance_mask = segmentation_id == segment_info['id']
                max_width, max_height = instance_mask.shape
                
                if min_x < 0 or max_x > max_width:
                    continue
                if min_y < 0 or max_y > max_height:
                    continue

                instance_mask = instance_mask[min_y:max_y, min_x:max_x]
                instance_mask_pil = Image.fromarray(instance_mask.astype(float)*256).convert('L')
                save_path = os.path.join(train_out_folder, str(instance_class), str(segment_info['id'])+'.png')
                instance_mask_pil.save(save_path)

    # valid instance mask generation (consider ones from train)
    for ann in train_coco_d['annotations']:
        if ann['file_name'].split('.')[0] not in val_selected_images:
            continue
        segmentation = np.array(
        Image.open(os.path.join(train_segmentations_folder, ann['file_name'])),
        dtype=np.uint8)
        segmentation_id = rgb2id(segmentation)
        for segment_info in ann['segments_info']:

            if segment_info['category_id'] < 91:
                
                instance_class = segment_info['category_id']
                x, y, width, height = segment_info['bbox']
                center_x, center_y = (x+(x+width))//2, (y+(y+height))//2
                box_length = max(width, height)

                if box_length < 32:
                    continue

                min_x, max_x = center_x - box_length//2, center_x + box_length//2
                min_y, max_y = center_y - box_length//2, center_y + box_length//2

                instance_mask = segmentation_id == segment_info['id']
                max_width, max_height = instance_mask.shape
                
                if min_x < 0 or max_x > max_width:
                    continue
                if min_y < 0 or max_y > max_height:
                    continue

                instance_mask = instance_mask[min_y:max_y, min_x:max_x]
                instance_mask_pil = Image.fromarray(instance_mask.astype(float)*256).convert('L')
                save_path = os.path.join(val_out_folder, str(instance_class), str(segment_info['id'])+'.png')
                instance_mask_pil.save(save_path)

    for ann in val_coco_d['annotations']:
        if ann['file_name'].split('.')[0] not in val_selected_images:
            continue
        segmentation = np.array(
        Image.open(os.path.join(val_segmentations_folder, ann['file_name'])),
        dtype=np.uint8)
        segmentation_id = rgb2id(segmentation)
        for segment_info in ann['segments_info']:

            if segment_info['category_id'] < 91:
                
                instance_class = segment_info['category_id']
                x, y, width, height = segment_info['bbox']
                center_x, center_y = (x+(x+width))//2, (y+(y+height))//2
                box_length = max(width, height)

                if box_length < 32:
                    continue

                min_x, max_x = center_x - box_length//2, center_x + box_length//2
                min_y, max_y = center_y - box_length//2, center_y + box_length//2

                instance_mask = segmentation_id == segment_info['id']
                max_width, max_height = instance_mask.shape
                
                if min_x < 0 or max_x > max_width:
                    continue
                if min_y < 0 or max_y > max_height:
                    continue

                instance_mask = instance_mask[min_y:max_y, min_x:max_x]
                instance_mask_pil = Image.fromarray(instance_mask.astype(float)*256).convert('L')
                save_path = os.path.join(val_out_folder, str(instance_class), str(segment_info['id'])+'.png')
                instance_mask_pil.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", type=str, default="./coco", help="path to coco dataset")
    parser.add_argument("--modified_coco_path", type=str, default="./modified_coco", help="path to save modified coco dataset")
    parser.add_argument("--out_path", type=str, default="./instance_mask", help="path to save instance mask")
    args = parser.parse_args()
    main(args)