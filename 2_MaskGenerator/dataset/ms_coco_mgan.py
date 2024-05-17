from locale import normalize
import os
import torch
import numpy as np

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image, ImageOps

import cv2
from matplotlib.patches import Rectangle

class MGANCOCODataset(Dataset):
    def __init__(self, set_name='train', transform=None, config_file = None):
        if config_file == None:
            self.set_name = set_name    
        else:
            self.config_file = config_file
            self.set_name = set_name
        
        self.data_dir = os.path.join(config_file['modified_coco_path'], 'annotations', 'instance_mask')
        
        self.transform = transform
        dataroot = os.path.join(self.data_dir, set_name+'2017')

        self.file_list = []
        self.class_list_string = [str(index) for index in self.config_file['category_id']]
        for (root, directories, file_names) in os.walk(dataroot):
            if root.split(set_name+'2017/')[-1] in self.class_list_string:
                for file_name in file_names:
                    if ('.png' in file_name):
                        file_path = os.path.join(root, file_name)
                        self.file_list.append(file_path)

        print(f"total_data_number : {len(self.file_list)}")
        print("finish loading file list")

    def gray_reader(self, image_path):
        im = Image.open(image_path)
        im2 = ImageOps.grayscale(im)
        im.close()
        return np.array(im2)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = self.gray_reader(self.file_list[index])
        label = int(self.file_list[index].split('/')[-2])
        # bbox = self.file_list[index].split('.')[0].split('_')[-6:-2]
        # bbox = [int(i) for i in bbox]
        # ori_width, ori_height = self.file_list[index].split('.')[0].split('_')[-2:]
        # ori_width, ori_height = int(ori_width), int(ori_height)
        # resized_bbox = self.normalize_box_xyxy(torch.tensor(bbox).float(), torch.tensor([ori_width, ori_height]).float())
        # resized_bbox = self.box_xyxy_to_cxcywh(resized_bbox)
        return self.transform(img), torch.tensor(label).long()#, resized_bbox

    def normalize_box_xyxy(self, boxes, ori_size):

        w, h = ori_size
        x1, y1, x2, y2 = boxes.unbind(-1)
        x1 /= w
        x2 /= w
        y1 /= h
        y2 /= h
        b = [x1, y1, x2, y2]

        return torch.stack(b, dim=-1)

    def box_xyxy_to_cxcywh(self, bboxes):

        x0, y0, x1, y1 = bboxes.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]

        return torch.stack(b, dim=-1)

    def box_cxcywh_to_xyxy(self, bboxes):
        x_c, y_c, w, h = bboxes.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]

        return torch.stack(b, dim=-1)

class MGANCOCODataModule(pl.LightningDataModule):
    def __init__(self, config_file=None):
        super().__init__()
        self.config_file = config_file
        self.batch_size = config_file['batch_size']
        self.num_workers = config_file['num_workers']

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.config_file['crop_size'], self.config_file['crop_size'])), transforms.Normalize([0.5], [0.5])])

    def setup(self, stage):
        self.COCO_train = MGANCOCODataset(set_name='train', transform=self.transform, config_file=self.config_file)
        # self.COCO_val = OGANCOCODataset(self.data_dir, set_name='val', transform=self.transform, config_file=self.config_file)
        # self.COCO_test = OGANCOCODataset(self.data_dir, set_name='val', transform=self.transform, config_file=self.config_file)

    def collate_fn(self, batch):
        zip_batch = zip(*batch)
        tuple_batch = tuple(zip_batch)
        return tuple_batch

    def train_dataloader(self):
        return DataLoader(self.COCO_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True) # ,collate_fn=self.collate_fn)
    def val_dataloader(self):
        return DataLoader(self.COCO_train, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.COCO_train, batch_size=self.batch_size, num_workers=self.num_workers)


