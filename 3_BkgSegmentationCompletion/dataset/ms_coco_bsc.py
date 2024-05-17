import os
from tkinter import Y
from cv2 import OPTFLOW_FARNEBACK_GAUSSIAN
import torch
import numpy as np

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageOps
import skimage.io as io


import json
import cv2
from matplotlib.patches import Rectangle

from panopticapi.utils import rgb2id
from dataset.transform import get_params, get_transform
from dataset.mask_generator_256 import RandomMask
from pathlib import Path



class BSCCOCODataset(Dataset):
    def __init__(self, set_name='train', transform=None, config_file = None):
        if config_file == None:
            self.set_name = set_name    
        else:
            self.config_file = config_file
            self.set_name = set_name

        self.coco_path = self.config_file['coco_path']
        self.modified_coco_path = self.config_file['modified_coco_path']

        self.transform = transform
        self.dataroot = os.path.join(self.coco_path, set_name+'2017')
        self.img_base_dir = os.path.join(self.coco_path, set_name+'2017')
        self.ann_base_dir = os.path.join(self.modified_coco_path, set_name+'2017')
        self.seg_base_dir = os.path.join(self.coco_path,'annotations', 'panoptic2seg', set_name+'2017')
        # self.mask_base_dir =  os.path.join(self.data_dir, 'val2017_mask4eval_.0~.25')
        # self.instance_base_dir = os.path.join(self.data_dir, 'instance', set_name+'2017')

        self.detr_detect_predict_dir = os.path.join(self.ann_base_dir, 'score_obj_quries_bboxes')
        self.detr_seg_predict_dir = os.path.join(self.ann_base_dir, 'predict_panoptic_seg')
        self._hole_range = [0, 1]
        self.json_full_file_name = os.path.join(self.coco_path,'annotations', f"panoptic_{set_name}2017.json")

        with open(self.json_full_file_name, "r") as file:
            self.file_annotations = json.load(file)['annotations']

        self.selected_images = list(set([i.split('_')[0] for i in os.listdir(os.path.join(self.ann_base_dir, 'image512'))]))
        self.refine_ann()
        if self.set_name == 'val':
            self.build_valid_mask()

    def refine_ann(self):
        refined_file_annotation = []
        if self.set_name == 'val':
            with open(os.path.join(self.coco_path, 'annotations/panoptic_train2017.json'), "r") as file:
                train_file_annotations = json.load(file)['annotations']
                self.file_annotations = train_file_annotations + self.file_annotations # some train images are used for validation set (include train annotations for these)

        for ann in self.file_annotations:
            file_base_name = ann['file_name'].split('.')[0]
            if file_base_name in self.selected_images:
                refined_file_annotation.append(ann)
        self.file_annotations = refined_file_annotation

    def build_valid_mask(self):
        print('Building validation mask...')
        self.valid_mask = {}

        for file_annotation in self.file_annotations:
            file_base_name = file_annotation['file_name'].split('.')[0]
            image_path = os.path.join(self.img_base_dir, file_base_name + '.jpg')
            if self.set_name == 'val' and not os.path.exists(image_path):
                image_path = image_path.replace('val2017', 'train2017')
            ori_img = Image.open(image_path).convert('RGB')
            params = get_params(opt=self.config_file, size=ori_img.size)
            transform_img = get_transform(self.config_file, params, train=(self.set_name=='val'))
            img = transform_img(ori_img)
            mask = RandomMask(img.shape[-1], hole_range=self._hole_range)
            self.valid_mask[file_base_name] = mask

    def preprocess_input(self, seg):
        _, h, w = seg.shape
        nc = self.config_file['label_nc']
        one_hot_label = torch.zeros((nc, h, w))
        one_hot_seg = one_hot_label.scatter_(0, seg.long(), 1.0)
        return one_hot_seg

    def __len__(self):
        return len(self.file_annotations)

    def __getitem__(self, index):

        file_annotation = self.file_annotations[index]
        file_base_name = file_annotation['file_name'].split('.')[0]

        image_path = os.path.join(self.img_base_dir, file_base_name + '.jpg')
        seg_path = os.path.join(self.seg_base_dir, file_base_name + '.png')

        if self.set_name == 'val' and not os.path.exists(image_path):
            image_path = image_path.replace('val2017', 'train2017')
            seg_path = seg_path.replace('val2017', 'train2017')

        ori_img = Image.open(image_path).convert('RGB')
        ori_seg = Image.open(seg_path).convert('L')

        params = get_params(opt=self.config_file, size=ori_seg.size)
        transform_seg = get_transform(self.config_file, params, method=Image.NEAREST, normalize=False, train=(self.set_name=='train'))
        seg = transform_seg(ori_seg)*255 ## To convert [0,1] pixel value to class label map

        transform_img = get_transform(self.config_file, params, train=(self.set_name=='train'))
        img = transform_img(ori_img)

        if self.set_name =='train':
            crop_size = self.config_file['crop_size']
            mask = RandomMask(img.shape[-1], hole_range=self._hole_range)
        else:
            mask = self.valid_mask[file_base_name]
            # ori_mask_path = Path(self.mask_base_dir) / (file_base_name + '.png')
            # mask = np.array(Image.open(ori_mask_path).convert('L'))/255
            # mask = mask[np.newaxis,:,:]

        img_mask = torch.from_numpy(np.repeat(mask, repeats=3, axis=0))
        seg_mask = torch.from_numpy(mask)

        white_img = torch.ones_like(img)
        missing_seg = torch.ones_like(seg)*(self.config_file['label_nc']+1)

        input_img = torch.where(img_mask.bool(), img, white_img)
        input_seg = torch.where(seg_mask.bool(), seg, missing_seg)
    
        input_dict = {
            'ori_img' : img, 
            'input_img' : input_img,
            'ori_seg' : seg.long(),
            'input_seg' : input_seg.long(),
            'seg_mask' : seg_mask,
        }

        return input_dict


class BSCCOCODataModule(pl.LightningDataModule):
    def __init__(self, config_file=None):
        super().__init__()
        self.config_file = config_file
        self.batch_size = config_file['batch_size']
        self.num_workers = config_file['num_workers']
        self.transform = None
        
    def setup(self, stage):
        self.COCO_train = BSCCOCODataset(set_name='train', transform=None, config_file=self.config_file)
        self.COCO_val = BSCCOCODataset(set_name='val', transform=None, config_file=self.config_file)
        self.COCO_test = BSCCOCODataset(set_name='val', transform=None, config_file=self.config_file)

    def collate_fn(self, batch):
        zip_batch = zip(*batch)
        tuple_batch = tuple(zip_batch)
        return tuple_batch

    def train_dataloader(self):
        return DataLoader(self.COCO_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True) # ,collate_fn=self.collate_fn)
    def val_dataloader(self):
        return DataLoader(self.COCO_val, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
    def test_dataloader(self):
        return DataLoader(self.COCO_test, batch_size=self.batch_size, num_workers=self.num_workers)