from platform import node
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as f
import torchvision
from torchvision import transforms

from model.interface import LitModel
import wandb
from PIL import Image
import copy

import matplotlib.pyplot as plt
from model.mit.transformer import ObjInferLayer_Type_1, ObjInferLayer_Type_2, ObjInferTransformer_type_1, ObjInferTransformer_type_2
from model.graph.gatn import GAT, GAT_res
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

class MIT(LitModel):
    def make_index_clsname_dict(self):
        import json
        import os

        json_path = os.path.join(self.config_file['coco_path'], 'annotations/panoptic_val2017.json')
        dataset = json.load(open(json_path))
        self.index2clsname = {0 : 'nothing'}
        self.clsname2index = {}
        self.index2isthing = {}
        self.total_index_list = []

        for category in dataset['categories']:
            self.total_index_list.append(category['id'])

        for category in dataset['categories']:
            self.index2clsname[category['id']] = category['name']

        for category in dataset['categories']:
            self.clsname2index[category['name']] = category['id']

        for category in dataset['categories']:
            self.index2isthing[category['id']] = category['isthing']

    def create_model(self):
        self.automatic_optimization = True
        self.make_index_clsname_dict()
        self.optim_config = self.config_file['optim_config']
        self.tf_config = self.config_file['mit_config']

        self.obj_inf_type = self.tf_config['obj_inf_model']
        self.no_pe = self.tf_config['obj_inf_no_pe']
        self.pe_type = self.tf_config['obj_inf_pe_type']

        detr_seg = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True).eval()
        self.fixed_classifier = copy.deepcopy(detr_seg.detr.class_embed)
        self.classifier = nn.Linear(256, 210)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=self.tf_config['smoothing'])
        
        if self.obj_inf_type == 'type1':
            enc_layer = ObjInferLayer_Type_1(d_model=self.tf_config['hidden_dim'], \
                nhead=self.tf_config['nhead'], \
                dim_feedforward=self.tf_config['dim_feedforward'], \
                dropout=self.tf_config['dropout'], \
                activation=self.tf_config['tr_enc_activation'], \
                normalize_before=self.tf_config['pre_norm']    
                )
            self.transformer = ObjInferTransformer_type_1(encoder_layer=enc_layer, num_layers=self.tf_config['enc_layers'], no_pe=self.no_pe, pe_type=self.pe_type, \
                no_classifier=self.tf_config['no_classifier'], dim=self.tf_config['hidden_dim'], class_numb=self.config_file['label_nc'], contain_is_thing=self.tf_config['contain_is_thing'])

        elif self.obj_inf_type == 'type2':
            enc_layer = ObjInferLayer_Type_2(d_model=self.tf_config['hidden_dim'], \
                nhead=self.tf_config['nhead'], \
                dim_feedforward=self.tf_config['dim_feedforward'], \
                dropout=self.tf_config['dropout'], \
                activation=self.tf_config['tr_enc_activation'], \
                normalize_before=self.tf_config['pre_norm']    
                )
            self.transformer = ObjInferTransformer_type_2(encoder_layer=enc_layer, num_layers=self.tf_config['enc_layers'], no_pe=self.no_pe, pe_type=self.pe_type, \
                no_classifier=self.tf_config['no_classifier'], dim=self.tf_config['hidden_dim'], class_numb=self.config_file['label_nc'], contain_is_thing=self.tf_config['contain_is_thing'])
        
        elif self.obj_inf_type == 'gat':
            self.transformer = GAT_res(channel_dim=[256+4, 128, 64, 256+4], heads_num=6)
            self.classifier = nn.Linear(256+4, 210)
        # self.automatic_optimization = False

        # from torchmetrics import Accuracy
        # top1_acc = Accuracy(num_classes=20, top_k=1)
        # top5_acc = Accuracy(num_classes=20, top_k=5)

    def configure_optimizers(self):

        if self.optim_config['optimizer'] == 'adam':
            optim = torch.optim.Adam(params=list(self.transformer.parameters()) + list(self.classifier.parameters()),
                                    lr=self.optim_config['lr_init'],
                                    betas=(self.optim_config['beta_1'], self.optim_config['beta_2']), weight_decay=self.optim_config['weight_decay'])

        elif self.optim_config['optimizer'] == 'sgd':
            optim = torch.optim.SGD(params=list(self.transformer.parameters()) + list(self.classifier.parameters()),
                                    lr=self.optim_config['lr_init'], weight_decay=self.optim_config['weight_decay'])


        if self.optim_config['scheduler'] == 'OneCycleLR':
            # import dataset.ms_coco_mit as ms_coco_mit
            # dataset = ms_coco_mit.MITCOCODataset('/root/data/MSCOCO2017', set_name='train', transform=None, config_file=self.config_file)
            # from torch.utils.data import Dataset, DataLoader, random_split
            # dataset_length = len(DataLoader(dataset, batch_size=self.config_file['batch_size'], num_workers=0, shuffle=True, drop_last=True))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.optim_config['lr_init'], pct_start=0.1, total_steps=500, anneal_strategy='linear')
        else:
            return [optim]

        return [optim], [scheduler]

    def gradient_switch(self, model: nn.Module, bool: bool):
        for p in model.parameters():
            p.requires_grad = bool

    def preprocess4gat(self, obj_quries, bboxes, missing_boxes, masks):
        graph_data_list = []
        for obj_feature, bbox, missing_bbox, mask in zip(obj_quries, bboxes, missing_boxes, masks):
            
            box_features = torch.cat([obj_feature, torch.normal(0, 1, size=(1, 256)).type_as(obj_feature)], dim=0).type_as(obj_feature)
            box_coordinates = torch.cat([bbox, missing_bbox.unsqueeze(0)], dim=0).type_as(obj_feature)
            node_features = torch.cat([box_features, box_coordinates], dim=1)
            node_features = node_features[[~mask]]
            edge_index_s = [i for i in range(0, node_features.shape[0]) for j in range(0, node_features.shape[0]-1)]
            edge_index_t = [j for i in range(0, node_features.shape[0]) for j in range(0, node_features.shape[0]) if i != j]

            graph_data_list.append(
                Data(
                    x=node_features, 
                    edge_index=torch.tensor([edge_index_s, edge_index_t]).type_as(node_features).type(torch.long)
                    )
                )
        
        graph_batch = Batch.from_data_list(graph_data_list)

        return graph_batch


    def training_step(self, batch, batch_idx):
        self.transformer.train()
        self.classifier.train()
        self.fixed_classifier.eval()
        # sch = self.lr_schedulers()
        # opt = self.optimizers()
        self.gradient_switch(self.transformer, True)
        self.gradient_switch(self.classifier, True)
        self.gradient_switch(self.fixed_classifier, False)

        key_mask_type_1, key_mask_type_2_sa, key_mask_type_2_ca = batch['key_mask_type_1'], batch['key_mask_type_2_sa'], batch['key_mask_type_2_ca']
        missing_box, missing_class = batch['missing_box'], batch['missing_class']
        ori_width, ori_height = batch['ori_width'], batch['ori_height']
        obj_queries, bboxes = batch['obj_quries'], batch['bboxes']

        missing_box = self.normalize_box_xyxy(missing_box, (ori_width, ori_height))
        missing_box = self.box_xyxy_to_cxcywh(missing_box)

        with torch.no_grad():
            obj_cls_logits = self.fixed_classifier(obj_queries)
            _, detected_obj_label = obj_cls_logits.softmax(-1)[..., :-1].max(-1)

            if self.tf_config['contain_is_thing']:
                is_thing_label = torch.zeros_like(detected_obj_label).float()
                for index in self.total_index_list:
                    is_thing_label[detected_obj_label==index] = self.index2isthing[index]
            else:
                is_thing_label = None

        if self.obj_inf_type == 'type1':
            missing_token, attn_matrix_list = self.transformer(detected_obj_label=detected_obj_label, bboxes=bboxes, missing_box=missing_box, key_padding_mask=key_mask_type_1, is_thing_label=is_thing_label)
        elif self.obj_inf_type == 'type2':
            missing_token, attn_matrix_list = self.transformer(detected_obj_label=detected_obj_label, bboxes=bboxes, missing_box=missing_box, sa_key_padding_mask=key_mask_type_2_sa, ca_key_padding_mask=key_mask_type_2_ca, is_thing_label=is_thing_label)
        elif self.obj_inf_type == 'gat':
            if self.tf_config['token_type'] == 'obj_query':
                obj_queries = self.transformer.label_embed(detected_obj_label)

            graph_batch = self.preprocess4gat(obj_queries, bboxes, missing_box, key_mask_type_1)
            missing_token = self.transformer(graph_batch)

        if self.tf_config['no_classifier']:
            final_embed = self.transformer.final_token_prediction(missing_token)
            missing_cls_logits = torch.matmul(missing_token, self.transformer.label_embd_layer.weight.T)# + self.transformer.final_bias
        else:
            missing_cls_logits = self.classifier(missing_token)

        cls_loss = self.ce_loss(missing_cls_logits.squeeze(dim=1), missing_class)

        # opt.zero_grad()
        # self.manual_backward(cls_loss)
        # opt.step()

        _, predict_label = missing_cls_logits.squeeze(dim=1).softmax(-1).max(-1)
        # self.visualization_matrix(attn_matrix_list=attn_matrix_list, mode='train', input_img=batch['input_img'].cpu(), detected_obj_label=detected_obj_label.cpu(), predict_label=predict_label.cpu(),
        #  missing_label=batch['missing_class'], mask=batch['key_mask_type_1'].cpu(), detailed_file_base_name=batch['file_name'])

        correct_count = (missing_class == predict_label).int().sum()
        cls_acc = (correct_count/missing_class.shape[0])*100

        self.log("train/{loss_name}".format(loss_name="cls_loss"), cls_loss, on_epoch=True, sync_dist=True)
        self.log("train/classification_accuracy", cls_acc, on_epoch=True, sync_dist=True)

        return cls_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.mit_predict_class_dict = {}

        self.transformer.eval()
        self.classifier.eval()
        self.fixed_classifier.eval()

        self.gradient_switch(self.transformer, False)
        self.gradient_switch(self.classifier, False)
        self.gradient_switch(self.fixed_classifier, False)


        key_mask_type_1, key_mask_type_2_sa, key_mask_type_2_ca = batch['key_mask_type_1'], batch['key_mask_type_2_sa'], batch['key_mask_type_2_ca']
        missing_box, missing_class = batch['missing_box'], batch['missing_class']
        ori_width, ori_height = batch['ori_width'], batch['ori_height']
        obj_queries, bboxes = batch['obj_quries'], batch['bboxes']
        
        missing_box = self.normalize_box_xyxy(missing_box, (ori_width, ori_height))
        missing_box = self.box_xyxy_to_cxcywh(missing_box)
        
        with torch.no_grad():
            obj_cls_logits = self.fixed_classifier(obj_queries)
            _, detected_obj_label = obj_cls_logits.softmax(-1)[..., :-1].max(-1)

            if self.tf_config['contain_is_thing']:
                is_thing_label = torch.zeros_like(detected_obj_label).float()
                for index in self.total_index_list:
                    is_thing_label[detected_obj_label==index] = self.index2isthing[index]
            else:
                is_thing_label = None

        if self.obj_inf_type == 'type1':
            missing_token, attn_matrix_list = self.transformer(detected_obj_label=detected_obj_label, bboxes=bboxes, missing_box=missing_box, key_padding_mask=key_mask_type_1, is_thing_label=is_thing_label)
        elif self.obj_inf_type == 'type2':
            missing_token, attn_matrix_list = self.transformer(detected_obj_label=detected_obj_label, bboxes=bboxes, missing_box=missing_box, sa_key_padding_mask=key_mask_type_2_sa, ca_key_padding_mask=key_mask_type_2_ca, is_thing_label=is_thing_label)
        elif self.obj_inf_type == 'gat':
            if self.tf_config['token_type'] == 'obj_query':
                obj_queries = self.transformer.label_embed(detected_obj_label)

            graph_batch = self.preprocess4gat(obj_queries, bboxes, missing_box, key_mask_type_1)
            missing_token = self.transformer(graph_batch)


        if self.tf_config['no_classifier']:
            final_embed = self.transformer.final_token_prediction(missing_token)
            missing_cls_logits = torch.matmul(missing_token, self.transformer.label_embd_layer.weight.T)# + self.transformer.final_bias
        else:
            missing_cls_logits = self.classifier(missing_token)

        cls_loss = self.ce_loss(missing_cls_logits.squeeze(dim=1), missing_class)

        _, predict_label = missing_cls_logits.squeeze(dim=1).softmax(-1).max(-1)
        # self.visualization_matrix(attn_matrix_list=attn_matrix_list, mode='val', input_img=batch['input_img'].cpu(), detected_obj_label=detected_obj_label.cpu(), predict_label=predict_label.cpu(),
        #  missing_label=batch['missing_class'], mask=batch['key_mask_type_1'].cpu(), detailed_file_base_name=batch['file_name'])
        correct_count = (missing_class == predict_label).int().sum()
        cls_acc = (correct_count/missing_class.shape[0])*100

        self.log("val/{loss_name}".format(loss_name="cls_loss"), cls_loss, on_epoch=True, sync_dist=True)
        self.log("Cls_Acc", cls_acc, on_epoch=True, sync_dist=True)
        for file_name, predict in zip(batch['file_name'], predict_label):
            self.mit_predict_class_dict[file_name] = predict.item()
            # print(predict.item())
            # print(file_name)

        return correct_count, predict_label.shape[0]

    def validation_epoch_end(self, outputs):
        import json
        
        with open('./ms_mit_predict_label.json', 'w') as outfile:
            json.dump(self.mit_predict_class_dict, outfile)

        total_correct_count = 0
        total_count = 0
        for correct, entire in outputs:
            total_correct_count += correct
            total_count += entire
        print("cls_acc", (total_correct_count/total_count)*100)

    def visualization_matrix_average(self, mode, attn_matrix_list, input_img, detected_obj_label, predict_label, missing_label, mask, detailed_file_base_name):
            import os
            os.makedirs(f'/home/jinoh/data/HVITA_Pytorch/attn_matrix/{mode}/avg_train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler', exist_ok=True)

            for i, attn_matrix in enumerate(attn_matrix_list):
                if i == 0:
                    average_attn_matrix = attn_matrix
                else:
                    average_attn_matrix += attn_matrix
            average_attn_matrix /= 12
            for image_idx in range(input_img.shape[0]):
                # fig, axs = plt.subplot_mosaic([['1', 'img'], ['2', 'img'], ['3', 'img'], ['4', 'img'], ['5', 'img'], ['6', 'img'], ['7', 'img'], ['8', 'img'], ['9', 'img'], ['10', 'img'], ['11', 'img'], ['12', 'img']], figsize=(15, 10))
                fig, axs = plt.subplot_mosaic([['1', 'img']], figsize=(15, 10))
                try: 
                    predict_str = self.index2clsname[predict_label[image_idx].item()]
                except KeyError:
                    predict_str = 'key_error'
                fig.suptitle(f'GT : {self.index2clsname[missing_label[image_idx].item()]}, Predict : {predict_str}', fontsize=16)
                for i, ax in enumerate(sorted(axs)):
                    if i < 1:
                        im = axs[ax].matshow(average_attn_matrix[image_idx,-1,:-1][~mask[image_idx][:100]].unsqueeze(0).cpu(),
                        vmin=torch.min(average_attn_matrix[image_idx,-1,:-1][~mask[image_idx][:100]]), 
                        vmax=torch.max(average_attn_matrix[image_idx,-1,:-1][~mask[image_idx][:100]]))
                        
                        obj_label = []
                        for class_index in detected_obj_label[image_idx][~mask[image_idx][:100]]:
                            obj_label.append(self.index2clsname[class_index.item()])

                        axs[ax].set_xticks(np.arange(len(detected_obj_label[image_idx][~mask[image_idx][:100]])), labels=obj_label)

                        axs[ax].set_yticks(np.arange(len(['missing'])), labels=['missing'])
                        
                        plt.setp(axs[ax].get_xticklabels(), rotation=45, ha="right",
                            rotation_mode="anchor")
                    else:
                        axs[ax].imshow((input_img[image_idx].permute(1,2,0)+1)*0.5)
                        # (ori_fake_images+1)*0.5).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

                    cax = plt.axes([0.48, 0.05, 0.5, 0.05])
                    fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=1.2)
                    fig.tight_layout()
                    fig.colorbar(im, cax=cax, orientation='horizontal')
                plt.savefig(f'/home/jinoh/data/HVITA_Pytorch/attn_matrix/{mode}/avg_train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler/{detailed_file_base_name[image_idx]}.png')
                plt.close()

    def visualization_matrix(self, mode, attn_matrix_list, input_img, detected_obj_label, predict_label, missing_label, mask, detailed_file_base_name):
        import os
        os.makedirs(f'/home/jinoh/data/HVITA_Pytorch/attn_matrix/{mode}/vg_train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler', exist_ok=True)
        attn_matrix_list = attn_matrix_list[0:-1:3]
            
        for image_idx in range(input_img.shape[0]):
            # fig, axs = plt.subplot_mosaic([['1', 'img'], ['2', 'img'], ['3', 'img'], ['4', 'img'], ['5', 'img'], ['6', 'img'], ['7', 'img'], ['8', 'img'], ['9', 'img'], ['10', 'img'], ['11', 'img'], ['12', 'img']], figsize=(15, 10))
            min = 1
            max = 0
            for i, attn_matrix in enumerate(attn_matrix_list):
                min_candidate = torch.min(attn_matrix_list[i][image_idx,-1,:-1][~mask[image_idx][:100]])
                max_candidate = torch.max(attn_matrix_list[i][image_idx,-1,:-1][~mask[image_idx][:100]])
                if min_candidate < min:
                    min = min_candidate
                if max_candidate > max:
                    max = max_candidate
            fig, axs = plt.subplot_mosaic([['1', 'img'], ['2', 'img'], ['3', 'img'], ['4', 'img']], figsize=(15, 10))
            try: 
                predict_str = self.index2clsname[predict_label[image_idx].item()]
            except KeyError:
                predict_str = 'key_error'
            fig.suptitle(f'GT : {self.index2clsname[missing_label[image_idx].item()]}, Predict : {predict_str}', fontsize=16)
            for layer_idx in range(len(attn_matrix_list)):
                for i, ax in enumerate(sorted(axs)):
                    if i < 4:
                        im = axs[ax].matshow(attn_matrix_list[layer_idx][image_idx,-1,:-1][~mask[image_idx][:100]].unsqueeze(0).cpu(),
                         vmin=min, 
                         vmax=max)
                        
                        obj_label = []
                        for class_index in detected_obj_label[image_idx][~mask[image_idx][:100]]:
                            obj_label.append(self.index2clsname[class_index.item()])

                        axs[ax].set_xticks(np.arange(len(detected_obj_label[image_idx][~mask[image_idx][:100]])), labels=obj_label)

                        axs[ax].set_yticks(np.arange(len(['missing'])), labels=['missing'])
                        
                        plt.setp(axs[ax].get_xticklabels(), rotation=45, ha="right",
                            rotation_mode="anchor")
                    else:
                        axs[ax].imshow((input_img[image_idx].permute(1,2,0)+1)*0.5)
                        # (ori_fake_images+1)*0.5).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

                cax = plt.axes([0.48, 0.05, 0.5, 0.05])
                fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=1.2)
                fig.tight_layout()
                fig.colorbar(im, cax=cax, orientation='horizontal')
            plt.savefig(f'/home/jinoh/data/HVITA_Pytorch/attn_matrix/{mode}/vg_train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler/{detailed_file_base_name[image_idx]}.png')
            plt.close()
            
    def normalize_box_xyxy(self, boxes, ori_size):

        #  boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
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