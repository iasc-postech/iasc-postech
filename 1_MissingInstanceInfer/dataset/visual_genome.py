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
from dataset.make_mask import MakeMaskMultipleBox, MakeMaskSingleBox


class MITCOCODataset(Dataset):
    def __init__(self, data_dir='/root/data/MSCOCO2017/', set_name='train', transform=None, config_file = None):
        if config_file == None:
            self.data_dir = data_dir
            self.set_name = set_name    
        else:
            self.config_file = config_file
            self.data_dir = config_file['data_dir']
            self.set_name = set_name

        self.transform = transform
        self.dataroot = os.path.join('/home/jinoh/data/VG', set_name+'2017')
        # self.img_base_dir = os.path.join(self.data_dir, set_name+'2017')
        self.img_base_dir = os.path.join('/home/jinoh/data/VG_100K')
        self.ann_base_dir = os.path.join('/home/jinoh/data/VG', 'toy_data4proposal_visual_genome', set_name+'2017')
        # self.seg_base_dir = os.path.join(self.data_dir, 'panoptic2seg', set_name+'2017')
        self.seg_base_dir = os.path.join('/home/jinoh/data/VG', 'panoptic2seg')
        # self.instance_base_dir = os.path.join(self.data_dir, 'instance', set_name+'2017')
        self.mat_base_dir = os.path.join(self.ann_base_dir, 'our_mat_result')

        self.detr_detect_predict_dir = os.path.join(self.ann_base_dir, 'score_obj_quries_bboxes')
        self.detr_seg_predict_dir = os.path.join(self.ann_base_dir, 'predict_panoptic_seg')
        self.mask_base_dir = os.path.join(self.ann_base_dir, 'mask512')
        json_full_file_name = os.path.join(self.ann_base_dir, 'toy_data4proposal.json')

        #######
        json_path = os.path.join('/home/jinoh/data/MSCOCO2017', 'panoptic/annotations/panoptic_val2017.json')
        dataset = json.load(open(json_path))
        self.index2clsname = {0 : 'nothing'}
        self.clsname2index = {}
        for category in dataset['categories']:
            self.index2clsname[category['id']] = category['name']
        for category in dataset['categories']:
            self.clsname2index[category['name']] = category['id']

        with open(json_full_file_name, "r") as file:
            self.file_list = json.load(file)
        self.file_list = self.file_list[:int(len(self.file_list))]

    def preprocess_input(self, seg):
        _, h, w = seg.shape
        nc = self.config_file['label_nc']
        one_hot_label = torch.zeros((nc, h, w))
        one_hot_seg = one_hot_label.scatter_(0, seg.long(), 1.0)
        return one_hot_seg

    def box_xyxy_to_cxcywh(self, bboxes):

        x0, y0, x1, y1 = bboxes.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]

        return torch.stack(b, dim=-1)


    def box_cxcywh_to_xyxy(self, bboxes):
        x_c, y_c, w, h = bboxes.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]

        return torch.stack(b, dim=-1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        file_annotation = self.file_list[index]
        missing_box = file_annotation['missing_bbox']
        missing_class = file_annotation['missing_class']

        file_base_name = file_annotation['score_obj_quries_bboxes'].split('_')[0]
        detailed_file_base_name = file_annotation['score_obj_quries_bboxes'].split('.')[0]


        image_path = os.path.join(self.img_base_dir, file_base_name + '.jpg')
        detr_detection_output_path = os.path.join(self.detr_detect_predict_dir, file_annotation['score_obj_quries_bboxes']+'.npz')
        score_obj_quries_bboxes = np.load(detr_detection_output_path, allow_pickle=True)
        score, obj_quries, bboxes = torch.tensor(score_obj_quries_bboxes['score']), torch.tensor(score_obj_quries_bboxes['obj_queries']), torch.tensor(score_obj_quries_bboxes['pred_boxes'])

        # print(score.shape, obj_quries.shape, bboxes.shape)
        mask = score < 0.8
        key_mask_type_1 = torch.cat([mask, torch.tensor([False])], dim=0)
        key_mask_type_2_sa = key_mask_type_2_ca = mask

        img = Image.open(image_path).convert('RGB')
        ori_width, ori_height = img.size

        x1, y1, x2, y2 = missing_box
        from PIL import ImageDraw
        import copy
        # masked_out_img = copy.deepcopy(img)
        # d = ImageDraw.Draw(masked_out_img)
        # d.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        x1, y1, x2, y2 = missing_box[0]*0.95/ori_width, missing_box[1]*0.95/ori_height, missing_box[2]*1.05/ori_width, missing_box[3]*1.05/ori_height
        if x2 >= 1:
            x2 = 1
        
        if y2 >= 1:
            y2 = 1



        if self.transform is None:
            from dataset.transform import get_params, get_transform
            from torchvision.utils import draw_bounding_boxes

            params = get_params(opt=self.config_file, size=img.size)
            transform_img = get_transform(self.config_file, params, train=(self.set_name=='train'))
            img = transform_img(img)
            # masked_out_img = transform_img(masked_out_img)
            resized_missing_box = (torch.tensor([x1, y1, x2, y2])*self.config_file['crop_size'])
            masked_out_img = img.clone()
            img_mask = MakeMaskMultipleBox((img.shape[-1],img.shape[-1]), bboxes=[resized_missing_box])
            mask_repeat = torch.from_numpy(np.repeat(img_mask, repeats=3, axis=0))
            white = torch.ones_like(masked_out_img)
            masked_out_img = torch.where(mask_repeat.bool(), masked_out_img, white)

            img_with_bboxes = torch.round((img*127.5 + 128).clamp(0.0, 255.0)).to(torch.uint8)
            img_with_bboxes = draw_bounding_boxes(img_with_bboxes, self.box_cxcywh_to_xyxy(bboxes[~mask])*self.config_file['crop_size'], colors="red", width=3)
    
        input_dict = {
            'ori_img' : img, 
            'input_img' : masked_out_img,
            'file_name' : detailed_file_base_name,
            'img_with_bboxes' : img_with_bboxes,
            'missing_class' : missing_class,
            'missing_box' : resized_missing_box,
            'obj_quries' : obj_quries,
            'bboxes' : bboxes, 
            'key_mask_type_1' : key_mask_type_1,
            'key_mask_type_2_sa' : key_mask_type_2_sa,
            'key_mask_type_2_ca' : key_mask_type_2_ca,
            'ori_width' : ori_width,
            'ori_height' : ori_height
        }

        return input_dict


class MITCOCODataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "", config_file=None):
        super().__init__()
        self.data_dir = data_dir
        self.config_file = config_file
        self.batch_size = config_file['batch_size']
        self.num_workers = config_file['num_workers']
        self.transform = None
        
    def setup(self, stage):
        self.COCO_train = MITCOCODataset(self.data_dir, set_name='val', transform=None, config_file=self.config_file)
        self.COCO_val = MITCOCODataset(self.data_dir, set_name='val', transform=None, config_file=self.config_file)
        self.COCO_test = MITCOCODataset(self.data_dir, set_name='val', transform=None, config_file=self.config_file)

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


