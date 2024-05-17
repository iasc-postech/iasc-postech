import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as f

from collections import OrderedDict

from model.mit.architecture import TransformerEncoder, TransformerEncoderLayer
from model.interface import LitModel
import utils.misc as misc
import wandb
import copy

class tr_enc(LitModel):
    def create_model(self):

        self.enc_layers =  self.config_file['enc_layers']
        self.dim_feedforward = self.config_file['dim_feedforward']
        self.d_model = self.config_file['hidden_dim']
        self.nhead = self.config_file['nhead']
        self.dropout = self.config_file['dropout']
        self.activation = self.config_file['tr_enc_activation']
        self.pre_norm = self.config_file['pre_norm']

        detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).eval()

        self.classifier = copy.deepcopy(detr.class_embed)

        encoder_layer = TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.activation, self.pre_norm)
        self.encoder_norm = nn.LayerNorm(self.d_model) if self.pre_norm else None
        self.tr_enc = TransformerEncoder(encoder_layer, num_layers=self.enc_layers, norm=self.encoder_norm)
        # self.automatic_optimization=False
        self.accumulation_step = self.config_file['accumulation_step']
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):

        if self.config_file['optimizer'] == 'adam':
            opt = torch.optim.Adam(params=self.tr_enc.parameters(),
                                    lr=self.config_file['lr_init'],
                                    weight_decay=self.config_file['weight_decay']
                                    )
        elif self.config_file['optimizer'] == 'adamw':
            opt = torch.optim.AdamW(params=self.tr_enc.parameters(),
                        lr=self.config_file['lr_init'],
                        weight_decay=self.config_file['weight_decay']
                                    )
        
        if self.config_file['scheduler'] is not None:
            if self.config_file['scheduler'] == 'lwca':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, 
                                                                 T_mult=2, eta_min=self.config_file['lr_init']*0.1)
            else:
                print(f"scheduler_is_{self.config_file['scheduler']}, not implemented yet")
                return [opt], []

            return [opt], [scheduler]
        else:
            return [opt], []

    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def training_step(self, batch, batch_idx):
        self.tr_enc.train()
        detr_query, attn_mask, key_padding_mask, detr_boxes, detr_rescaled_boxes, detr_predict_classes, missing_boxes, missing_classes = batch
        
        output = self.tr_enc(detr_query, detr_rescaled_boxes, missing_boxes, None, attn_mask)
        output = output.permute(1,0,2)[:,-1,:]
        class_logits = self.classifier(output)
        cls_loss = self.ce_loss(class_logits, missing_classes)
        self.log("train/{loss_name}".format(loss_name="cls_loss"), cls_loss.item(), on_step=True, logger=True)

        if np.isnan(cls_loss.item()):
            import sys
            print("model diverge")
            sys.exit(0)

        return cls_loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.tr_enc.eval()
        detr_query, attn_mask, key_padding_mask, detr_boxes, detr_rescaled_boxes, detr_predict_classes, missing_boxes, missing_classes = batch
        
        output = self.tr_enc(detr_query, detr_rescaled_boxes, missing_boxes, None, attn_mask)
        output = output.permute(1,0,2)[:,-1,:]
        class_logits = self.classifier(output)
        predict_class = class_logits.softmax(-1).argmax(-1)
        gt_class = missing_classes

        return predict_class, gt_class

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        accum_prediction_list = []
        accum_target_list = []

        for output in outputs:
            accum_prediction_list.append(output[0])
            accum_target_list.append(output[1])

        predictions = torch.cat(accum_prediction_list, dim=0)
        targets = torch.cat(accum_target_list, dim=0)
        
        total_pre_num = predictions.shape[0]
        correct_pred = 0
        for gt, prediction in zip(targets, predictions):
            if gt == prediction:
                correct_pred += 1

        accuracy = round(correct_pred/total_pre_num*100, 3)
        print("Classfication Accuracy: {acc}".format(acc=accuracy))
        self.log('val/acc', accuracy, logger=True)
        self.log('acc', accuracy, logger=True)

        return True