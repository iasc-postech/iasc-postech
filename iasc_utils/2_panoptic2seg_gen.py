import os, sys
import numpy as np
import json

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from panopticapi.utils import IdGenerator, rgb2id


json_file = './coco/annotations/panoptic_val2017.json'
segmentations_folder = './coco/annotations/panoptic_val2017'
img_folder = './coco/val2017/'
panoptic_coco_categories = './coco/annotations/panoptic_coco_categories.json'
out_folder = './test_panoptic2seg/val2017'

with open(json_file, 'r') as f:
    coco_d = json.load(f)

with open(panoptic_coco_categories, 'r') as f:
    categories_list = json.load(f)
categegories = {category['id']: category for category in categories_list}


os.makedirs(out_folder, exist_ok=True)

for ann in coco_d['annotations'][:100]:
    segmentation = np.array(
    Image.open(os.path.join(segmentations_folder, ann['file_name'])),
    dtype=np.uint8)
    segmentation_id = rgb2id(segmentation)
    for segment_info in ann['segments_info']:
            
        segments_class = segment_info['category_id']
        temp_seg = np.empty_like(segmentation_id)
        temp_seg[segmentation_id == segment_info['id']] = segments_class

    temp_seg_pil = Image.fromarray(temp_seg.astype(float)).convert('L')
    save_path = os.path.join(out_folder, ann['file_name'])
    temp_seg_pil.save(save_path)