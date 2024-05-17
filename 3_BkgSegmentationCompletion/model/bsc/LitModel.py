
from pandas import merge_ordered
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
from torchvision import transforms
from collections import OrderedDict
import numpy as np

from model.bsc.architecture import BidirectionalTransformer_avgpool, BidirectionalTransformer_sconv, BidirectionalSwinTransformer_sconv
from model.bsc.my_transformer import BidirectionalTransformer_local_global, BidirectionalTransformer_local_global_mask, BidirectionalTransformer_swin_mask
from model.bsc.unet_model import UNet
from model.interface import LitModel
from mIoU import AverageMeter, intersectionAndUnionGPU

import wandb
import copy
from PIL import Image
import cv2
from einops import rearrange, repeat
# from mmseg.core import mean_iou
from metric import mIoU

## from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/vis.py
def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def color_seg(seg, palette):
    """
    Replace classes with their colors.
    Takes:
        seg: H x W segmentation image of class IDs
    Gives:
        H x W x 3 image of class colors
    """
    return palette[seg.flat].reshape(seg.shape + (3,))

class BSC(LitModel):

    def make_index_clsname_dict(self):
        import json
        import os

        json_path = os.path.join(self.config_file['coco_path'], 'annotations/panoptic_val2017.json')
        dataset = json.load(open(json_path))
        self.index2clsname = {}
        self.clsname2index = {}

        for category in dataset['categories']:
            self.index2clsname[category['id']] = category['name']

        for category in dataset['categories']:
            self.clsname2index[category['name']] = category['id']

    def create_model(self):
        self.automatic_optimization = True
        import os

        self.optim_config = self.config_file['optim_config']
        self.tf_config = self.config_file['tf_config']
        self.batch_size = self.config_file['batch_size']
        self.label_nc = self.config_file['label_nc']
        self.ignore_index = 255

        self.make_index_clsname_dict()

        if self.tf_config['type'] == 'swin':
            self.bidi_trans = BidirectionalSwinTransformer_sconv(self.tf_config, label_nc=self.label_nc) 
        elif self.tf_config['type'] == 'vanilla':
            self.bidi_trans = BidirectionalTransformer_sconv(self.tf_config, label_nc=self.label_nc) 
        elif self.tf_config['type'] == 'unet':
            self.bidi_trans = UNet(self.tf_config, label_nc=self.label_nc) 
        print(self.bidi_trans)
        # BidirectionalTransformer_swin_mask(self.tf_config)
        # BidirectionalTransformer_sconv(self.tf_config) 
        # BidirectionalTransformer_avgpool(self.tf_config)
        # BidirectionalSwinTransformer(self.tf_config)
        # BidirectionalTransformer(self.tf_config)

        # bidi_trans_dict = torch.load(self.config_file['checkpoint_path'])
        # new_state_dict_3 = OrderedDict()
        # for key_name in bidi_trans_dict['state_dict'].keys():
        #     if key_name.split('.')[0] == 'bidi_trans':
        #         new_state_dict_3[key_name.split('bidi_trans.')[-1]] = bidi_trans_dict['state_dict'][key_name]

        # self.bidi_trans.load_state_dict(new_state_dict_3)


        self.palette = make_palette(num_classes=self.label_nc+2)
        self.cls_ce_loss = torch.nn.CrossEntropyLoss()
        self.mIoU_train = mIoU(class_numb=self.label_nc, ignore_index=self.ignore_index)
        self.mIoU_val = mIoU(class_numb=self.label_nc, ignore_index=self.ignore_index)
        self.mIoU_test = mIoU(class_numb=self.label_nc, ignore_index=self.ignore_index)


    def configure_optimizers(self):

        if self.optim_config['optimizer'] == 'adam':
            opt = torch.optim.Adam(params=list(self.bidi_trans.parameters()),
                                    lr=self.optim_config['lr_init'],
                                    betas=(self.optim_config['beta_1'], self.optim_config['beta_2']))
        elif self.optim_config['optimizer'] == 'sgd':
            opt = torch.optim.SGD(params=list(self.bidi_trans.parameters()),
                                    lr=self.optim_config['lr_init'])
        return [opt], []

    def gradient_switch(self, model: nn.Module, bool: bool):
        for p in model.parameters():
            p.requires_grad = bool

    def training_step(self, batch, batch_idx):

        self.bidi_trans.train()
        self.gradient_switch(self.bidi_trans, True)
        input_img, input_seg = batch['input_img'], batch['input_seg']
        ori_img, ori_seg = batch['ori_img'], batch['ori_seg']
        final_logits = self.bidi_trans(input_seg.squeeze())
        final_ce_loss = f.cross_entropy(final_logits.permute(0,3,1,2), ori_seg.squeeze().long())
        final_logits = final_logits[:,:,:,:final_logits.shape[-1]-1]
        _, predict_seg = torch.max(final_logits, dim=-1)
        loss = final_ce_loss
        self.log('train/ce_loss', loss.item(),  on_step=True, logger=True)

        if batch_idx % 100 == 0:

            self.log_img(input_seg.squeeze(), predict_seg.squeeze(), ori_seg.squeeze(), mode='train')
        
        predict_seg_mIoU, ori_seg_mIoU = predict_seg.squeeze().clone().detach(), ori_seg.squeeze().clone().detach()
        mask = batch['seg_mask'].squeeze().bool().clone() ## mask for only missing region
        predict_seg_mIoU[mask], ori_seg_mIoU[mask] = self.ignore_index, self.ignore_index
        self.mIoU_train.update(predict_seg_mIoU, ori_seg_mIoU)

        return loss 

    def training_epoch_end(self, outputs):

        self.train_mIoU = self.mIoU_train.compute()
        self.log('train_mIoU', self.train_mIoU)
        self.mIoU_train.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        self.bidi_trans.eval()
        self.gradient_switch(self.bidi_trans, False)
        input_img, input_seg = batch['input_img'], batch['input_seg']
        ori_img, ori_seg = batch['ori_img'], batch['ori_seg']
        final_logits = self.bidi_trans(input_seg.squeeze())
        final_ce_loss = f.cross_entropy(final_logits.permute(0,3,1,2), ori_seg.squeeze().long())
        final_logits = final_logits[:,:,:,:final_logits.shape[-1]-1]
        _, predict_seg = torch.max(final_logits, dim=-1)
        loss = final_ce_loss
        self.log('val/ce_loss', loss.item(),  on_step=True, logger=True)

        if batch_idx % 100 == 0:

            self.log_img(input_seg.squeeze(), predict_seg.squeeze(), ori_seg.squeeze(), mode='val')
        
        predict_seg_mIoU, ori_seg_mIoU = predict_seg.squeeze().clone().detach(), ori_seg.squeeze().clone().detach()
        mask = batch['seg_mask'].squeeze().bool().clone() ## mask for only missing region
        predict_seg_mIoU[mask], ori_seg_mIoU[mask] = self.ignore_index, self.ignore_index
        self.mIoU_val.update(predict_seg_mIoU, ori_seg_mIoU)

    @torch.no_grad()
    def validation_epoch_end(self, outputs):

        self.valid_mIoU = self.mIoU_val.compute()
        self.log('valid_mIoU', self.valid_mIoU)
        self.mIoU_val.reset()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        self.bidi_trans.eval()
        self.gradient_switch(self.bidi_trans, False)
        input_img, input_seg = batch['input_img'], batch['input_seg']
        ori_img, ori_seg = batch['ori_img'], batch['ori_seg']
        final_logits = self.bidi_trans(input_seg.squeeze())
        final_ce_loss = f.cross_entropy(final_logits.permute(0,3,1,2), ori_seg.squeeze().long())
        final_logits = final_logits[:,:,:,:final_logits.shape[-1]-1]
        _, predict_seg = torch.max(final_logits, dim=-1)
        
        predict_seg_mIoU, ori_seg_mIoU = predict_seg.squeeze().clone().detach(), ori_seg.squeeze().clone().detach()
        mask = batch['seg_mask'].squeeze().bool().clone() ## mask for only missing region
        predict_seg_mIoU[mask], ori_seg_mIoU[mask] = self.ignore_index, self.ignore_index
        self.mIoU_test.update(predict_seg_mIoU, ori_seg_mIoU)
        final_predict_seg = torch.where(mask, ori_seg.squeeze(), predict_seg)
        self.save_seg_img(input_seg.squeeze(), final_predict_seg.squeeze(), ori_seg.squeeze(), batch_idx)

    @torch.no_grad()
    def test_epoch_end(self, outputs):

        self.test_mIoU = self.mIoU_test.compute()
        self.log('test_mIoU', self.test_mIoU)
        self.mIoU_test.reset()
    
    def save_seg_img(self, input_seg, predict_seg, gt_seg, batch_idx):
        input_seg = self.all_gather(input_seg).flatten(0,1).detach().cpu().numpy()
        predict_seg = self.all_gather(predict_seg).flatten(0,1).detach().cpu().numpy()
        gt_seg = self.all_gather(gt_seg).flatten(0,1).detach().cpu().numpy()

        input_seg_img = color_seg(input_seg, self.palette)
        predict_seg_img = color_seg(predict_seg, self.palette)
        gt_seg_img = color_seg(gt_seg, self.palette)
        
        totensor = transforms.ToTensor()


        import matplotlib.pyplot as plt
        # if input_seg_img.shape[0] > 16:
        #     input_seg_img = input_seg_img[:16]
        #     predict_seg_img = predict_seg_img[:16]
        #     gt_seg_img = gt_seg_img[:16]

        # else:
        #     pass
        # ncols, nrows = 4, 4
        # figure, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(32,32))
        from torchvision.utils import draw_bounding_boxes, save_image
        import os
        path = '/home/jinoh/data/MSCOCO2017/toy_data4proposal3_0907_new_class/val2017/seg_restored_unet_result'
        os.makedirs(path, exist_ok=True)

        for i, (input, predict, gt) in enumerate(zip(input_seg_img, predict_seg_img, gt_seg_img)):
            visualize_list = []
            visualize_list.append(totensor(input))
            visualize_list.append(totensor(predict))
            visualize_list.append(totensor(gt))
            # grid_samples = torchvision.utils.make_grid(visualize_list, nrow=3).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8).to('cpu')
            # grid_samples = torchvision.utils.make_grid(visualize_list, nrow=3).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            # x = i//nrows
            # y = i%nrows 
            # ax = axes[x,y]
            # ax.imshow(grid_samples)
            save_image(visualize_list, os.path.join(path, str(self.batch_size*batch_idx+i)+'.png'))

        # self.logger.experiment.log({
        #     f"test_sample/input_pred_gt": figure
        #     })

        # plt.close(figure)
        
        

    def log_img(self, input_seg, predict_seg, gt_seg, mode='train'):
        input_seg = self.all_gather(input_seg).flatten(0,1).detach().cpu().numpy()
        predict_seg = self.all_gather(predict_seg).flatten(0,1).detach().cpu().numpy()
        gt_seg = self.all_gather(gt_seg).flatten(0,1).detach().cpu().numpy()

        input_seg_img = color_seg(input_seg, self.palette)
        predict_seg_img = color_seg(predict_seg, self.palette)
        gt_seg_img = color_seg(gt_seg, self.palette)
        totensor = transforms.ToTensor()

        import matplotlib.pyplot as plt
        if input_seg_img.shape[0] > 16:
            input_seg_img = input_seg_img[:16]
            predict_seg_img = predict_seg_img[:16]
            gt_seg_img = gt_seg_img[:16]

        else:
            pass
        ncols, nrows = 4, 4
        figure, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(32,32))

        for i, (input, predict, gt) in enumerate(zip(input_seg_img, predict_seg_img, gt_seg_img)):
            visualize_list = []
            visualize_list.append(totensor(input))
            visualize_list.append(totensor(predict))
            visualize_list.append(totensor(gt))
            grid_samples = torchvision.utils.make_grid(visualize_list, nrow=3).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            x = i//nrows
            y = i%nrows 
            ax = axes[x,y]
            ax.imshow(grid_samples)

        self.logger.experiment.log({
            f"{mode}_sample/input_pred_gt": figure
            })

        plt.close(figure)