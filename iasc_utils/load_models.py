import yaml
import os
import sys
import pickle
import torch

def load_mit(checkpoint_path):
    sys.path.append("./1_MissingInstanceInfer")
    from model.mit.transformer import ObjInferLayer_Type_1, ObjInferLayer_Type_2, ObjInferTransformer_type_1, ObjInferTransformer_type_2

    config_path = "./1_MissingInstanceInfer/configs/type_1_abs4c_no_classifier_deep.yaml"
    with open(config_path, 'r') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)
    
    tf_config = config_file['mit_config']
    obj_inf_type = tf_config['obj_inf_model']
    no_pe = tf_config['obj_inf_no_pe']
    pe_type = tf_config['obj_inf_pe_type']    

    if obj_inf_type == 'type1':
            enc_layer = ObjInferLayer_Type_1(d_model=tf_config['hidden_dim'], \
                nhead=tf_config['nhead'], \
                dim_feedforward=tf_config['dim_feedforward'], \
                dropout=tf_config['dropout'], \
                activation=tf_config['tr_enc_activation'], \
                normalize_before=tf_config['pre_norm']    
                )
            transformer = ObjInferTransformer_type_1(encoder_layer=enc_layer, num_layers=tf_config['enc_layers'], no_pe=no_pe, pe_type=pe_type, \
                no_classifier=tf_config['no_classifier'], dim=tf_config['hidden_dim'], class_numb=config_file['label_nc'], contain_is_thing=tf_config['contain_is_thing'])

    elif obj_inf_type == 'type2':
        enc_layer = ObjInferLayer_Type_2(d_model=tf_config['hidden_dim'], \
            nhead=tf_config['nhead'], \
            dim_feedforward=tf_config['dim_feedforward'], \
            dropout=tf_config['dropout'], \
            activation=tf_config['tr_enc_activation'], \
            normalize_before=tf_config['pre_norm']    
            )
        transformer = ObjInferTransformer_type_2(encoder_layer=enc_layer, num_layers=tf_config['enc_layers'], no_pe=no_pe, pe_type=pe_type, \
            no_classifier=tf_config['no_classifier'], dim=tf_config['hidden_dim'], class_numb=config_file['label_nc'], contain_is_thing=tf_config['contain_is_thing'])
    
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = {}

    for key in checkpoint['state_dict'].keys():
        if 'transformer' in key:
            new_state_dict[key.replace('transformer.', '')] = checkpoint['state_dict'][key]

    transformer.load_state_dict(new_state_dict)
    return transformer

def load_mgan(checkpoint_path):
    sys.path.append("./2_MaskGenerator")
    from model.mgan.architecture import Generator, Discriminator, MODULES
    config_path = "./2_MaskGenerator/configs/default_no_attn_feature_condition_false.yaml"
    with open(config_path, 'r') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)
    mgan_config = config_file['mgan_config']
    num_classes = config_file['label_nc']
    module = MODULES(apply_g_sn=mgan_config['apply_g_sn'], apply_d_sn=mgan_config['apply_d_sn'])

    generator = Generator(
        z_dim=mgan_config['z_dim'],
        g_shared_dim=mgan_config['g_shared_dim'],
        img_size=config_file['crop_size'],
        g_conv_dim=mgan_config['g_conv_dim'],
        apply_attn=mgan_config['apply_attn'],
        attn_g_loc=mgan_config['attn_g_loc'],
        num_classes=num_classes,
        MODULES=module,
        extra_feature_condition=mgan_config['extra_feature_condition']
        )
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = {}
    checkpoint['state_dict'].keys()
    for key in checkpoint['state_dict'].keys():
        if 'generator' in key:
            new_state_dict[key.replace('generator.', '')] = checkpoint['state_dict'][key]

    generator.load_state_dict(new_state_dict)
    return generator

def load_bsc(checkpoint_path):
    sys.path.append("./3_BkgSegmentationCompletion")
    from model.bsc.architecture import BidirectionalTransformer_avgpool, BidirectionalTransformer_sconv, BidirectionalSwinTransformer_sconv
    from model.bsc.my_transformer import BidirectionalTransformer_local_global, BidirectionalTransformer_local_global_mask, BidirectionalTransformer_swin_mask
    from model.bsc.unet_model import UNet

    config_path = "./3_BkgSegmentationCompletion/configs/vanilla.yaml"
    with open(config_path, "r") as fp: 
        config_file = yaml.load(fp, Loader=yaml.FullLoader)
    
    tf_config = config_file['tf_config']
    label_nc = config_file['label_nc']
    if tf_config['type'] == 'swin':
        bidi_trans = BidirectionalSwinTransformer_sconv(tf_config, label_nc=label_nc) 
    elif tf_config['type'] == 'vanilla':
        bidi_trans = BidirectionalTransformer_sconv(tf_config, label_nc=label_nc) 
    elif tf_config['type'] == 'unet':
        bidi_trans = UNet(tf_config, label_nc=label_nc)
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = {}
    for key in checkpoint['state_dict'].keys():
        if 'bidi_trans' in key:
            new_state_dict[key.replace('bidi_trans.', '')] = checkpoint['state_dict'][key]
    bidi_trans.load_state_dict(new_state_dict)
    return bidi_trans
