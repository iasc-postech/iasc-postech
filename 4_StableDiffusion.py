import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import dill
import torchvision
import torchvision.transforms as transforms
import iasc_utils.misc as misc
from iasc_utils.load_models import load_mit, load_mgan, load_bsc
from PIL import Image
from tqdm import tqdm
import json
import os
import argparse
import random
import copy

def disable_safety_checker(image, device, dtype):
    return image, None

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def normalize_box_xyxy(boxes, ori_size):

    #  boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    w, h = ori_size

    x1, y1, x2, y2 = boxes.unbind(-1)

    x1 /= w
    x2 /= w
    y1 /= h
    y2 /= h

    b = [x1, y1, x2, y2]

    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(bboxes):

    x0, y0, x1, y1 = bboxes.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
        (x1 - x0), (y1 - y0)]

    return torch.stack(b, dim=-1)

def main(args):
    fix_seed(args.seed)
    index2clsname = {0 : 'nothing'}

    base_image_dir = os.path.join(args.modified_coco_path, 'val2017/image512')
    base_mask_dir = os.path.join(args.modified_coco_path, 'val2017/mask512')
    base_detect_predict_dir = os.path.join(args.modified_coco_path, 'val2017/score_obj_quries_bboxes')
    base_seg_predict_dir = os.path.join(args.modified_coco_path, 'val2017/predict_panoptic_seg')

    label_info_path = os.path.join(args.coco_path, 'annotations/panoptic_val2017.json')
    label_info = json.load(open(label_info_path))
    for category in label_info['categories']:
        index2clsname[category['id']] = category['name']

    ann_file_name = os.path.join(args.modified_coco_path, 'val2017/modified_data.json')
    with open(ann_file_name, "r") as file:
        file_list = json.load(file)

    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
    pipe.run_safety_checker = disable_safety_checker
    pipe = pipe.to("cuda")

    detr_pseg, postprocessor_pseg = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True,  return_postprocessor=True)
    detr_pseg = detr_pseg.to("cuda")
    detr_pseg.eval()
    fixed_classifier = copy.deepcopy(detr_pseg.detr.class_embed)

    mit = load_mit(args.mit_ckpt_path)
    mask_generator = load_mgan(args.mgan_ckpt_path)
    bidi_trans = load_bsc(args.bsc_ckpt_path)

    mit = mit.to("cuda")
    bidi_trans = bidi_trans.to("cuda")
    mask_generator = mask_generator.to("cuda")

    mit.eval()
    bidi_trans.eval()
    mask_generator.eval()


    os.makedirs(os.path.join(args.output_path, 'init_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'images'), exist_ok=True)

    @torch.no_grad()
    def mgan_predict(labels):
        z = torch.randn(labels.shape[0], mask_generator.z_dim).to(labels.get_device())
        mask_generator.apply(misc.untrack_bn_statistics)
        fake_mask = mask_generator(z=z, label=labels, node_feature=None)
        return fake_mask

    @torch.no_grad()
    def bsc_predict(input_seg):
        final_logits = bidi_trans(input_seg.squeeze().unsqueeze(0))
        final_logits = final_logits[:,:,:,:final_logits.shape[-1]-1]
        _, predict_seg = torch.max(final_logits, dim=-1)
        return predict_seg

    def insert_mask(detr_seg_batch, missing_class_batch, mr_coordinate_batch, generated_mask_batch):
        # H, W = detr_seg_batch.shape[-2], detr_seg_batch.shape[-1]
        inserted_detr_seg_list = []
        for detr_seg, missing_class, mr_coordinate, generated_mask in zip(detr_seg_batch, missing_class_batch, mr_coordinate_batch, generated_mask_batch):
            x1, y1, x2, y2 = int(mr_coordinate[0]),  int(mr_coordinate[1]),  int(mr_coordinate[2]),  int(mr_coordinate[3])
            generated_mask = torch.where(generated_mask>0, True, False)
            generated_mask_resize = torchvision.transforms.Resize((y2-y1, x2-x1), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)(generated_mask)
            mask = torch.zeros_like(detr_seg).bool()
            mask, generated_mask_resize = mask.squeeze(), generated_mask_resize.squeeze()
            mask[y1:y2, x1:x2] = generated_mask_resize

            inserted_detr_seg = torch.where(mask.unsqueeze(0)==True, missing_class, detr_seg.long())
            inserted_detr_seg_list.append(inserted_detr_seg)

        inserted_detr_seg_batch = torch.cat(inserted_detr_seg_list, dim=0)
        return inserted_detr_seg_batch 

    for file_info in tqdm(file_list[int(len(file_list)*args.start):int(len(file_list)*args.end)]):
        file_name = file_info['score_obj_quries_bboxes']
        image_path = os.path.join(base_image_dir, file_name+'.png')
        mask_path = os.path.join(base_mask_dir, file_name+'.png')
        detr_detection_output_path = os.path.join(base_detect_predict_dir, file_name+'.npz')
        detr_seg_output_path = os.path.join(base_seg_predict_dir, file_name+'.npz')
            
        init_image = PIL.Image.open(image_path).convert("RGB")
        mask_image = PIL.Image.open(mask_path).crop((10,10,502,502)).resize((512,512),resample=PIL.Image.NEAREST)
        seg_mask_image = PIL.Image.open(mask_path).crop((10,10,502,502)).resize((256,256),resample=PIL.Image.NEAREST)

        mask_arry = np.array(mask_image)
        seg_mask_arry = np.array(seg_mask_image)
        bool_mask = torch.tensor(mask_arry[np.newaxis,:,:]).bool()

        mask_max = 255
        mask_arry = mask_max - mask_arry
        inverted_mask_image = PIL.Image.fromarray(mask_arry).convert("RGB")

        mask = torch.tensor(mask_arry[np.newaxis,:,:]).bool()
        seg_mask = torch.tensor(seg_mask_arry[np.newaxis,:,:]).bool()

        ## DeTR Predict Load
        score_obj_quries_bboxes = np.load(detr_detection_output_path, allow_pickle=True)
        score, obj_quries, bboxes = torch.tensor(score_obj_quries_bboxes['score']), torch.tensor(score_obj_quries_bboxes['obj_queries']), torch.tensor(score_obj_quries_bboxes['pred_boxes'])

        key_mask = score < 0.8
        key_mask_type_1 = torch.cat([key_mask, torch.tensor([False])], dim=0)
        key_mask = key_mask.to('cuda')
        key_mask_type_1 = key_mask_type_1.to('cuda')

        missing_box = masks_to_boxes(torch.bitwise_not(bool_mask))
        missing_box = missing_box.squeeze()
        x1, y1, x2, y2 = missing_box[0], missing_box[1], missing_box[2], missing_box[3]
        resized_missing_box = torch.tensor([x1, y1, x2, y2]).float()
        ori_width, ori_height = 512, 512
        resized_missing_box = normalize_box_xyxy(resized_missing_box, (ori_width, ori_height))
        resized_missing_box = box_xyxy_to_cxcywh(resized_missing_box)
        resized_missing_box = resized_missing_box.to('cuda')
        is_thing_label = None

        ## DeTR Seg Load
        detr_predict_seg_np = np.load(detr_seg_output_path, allow_pickle=True)
        detr_predict_seg_id, detr_predict_seg_info = detr_predict_seg_np['seg'], detr_predict_seg_np['seg_info']
        for seg_info in detr_predict_seg_info:
            instance_id = seg_info['id']
            detr_predict_seg_id[detr_predict_seg_id==instance_id] = seg_info['category_id']
        detr_predict_seg = detr_predict_seg_id
        detr_predict_seg = Image.fromarray(detr_predict_seg)


        transform_seg_list = [transforms.Resize(256, interpolation=Image.NEAREST), transforms.ToTensor()]
        transform_seg = transforms.Compose(transform_seg_list)
        detr_predict_seg_tensor = transform_seg(detr_predict_seg).long() ## Do not multiply 255
        missing_seg = (torch.ones_like(detr_predict_seg_tensor)*(200+1)).long()
        input_seg = torch.where(seg_mask.bool(), detr_predict_seg_tensor, missing_seg)
        input_seg = input_seg.to('cuda')
        

        obj_quries = obj_quries.to('cuda')
        bboxes = bboxes.to('cuda')

        obj_cls_logits = fixed_classifier(obj_quries)
        _, detected_obj_label = obj_cls_logits.softmax(-1)[..., :-1].max(-1)


        missing_token, attn_matrix_list = mit(detected_obj_label=detected_obj_label.unsqueeze(0), bboxes=bboxes.unsqueeze(0), missing_box=resized_missing_box.unsqueeze(0), key_padding_mask=key_mask_type_1.unsqueeze(0), is_thing_label=is_thing_label)
        missing_cls_logits = torch.matmul(missing_token, mit.label_embd_layer.weight.T)
        _, predict_label = missing_cls_logits.squeeze(dim=1).softmax(-1).max(-1)

        instance_mask = mgan_predict(labels=predict_label)
        recon_seg = bsc_predict(input_seg)
        recon_seg = torchvision.transforms.Resize((512, 512), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)(recon_seg)
        recon_final_seg = insert_mask(recon_seg, predict_label.unsqueeze(0), missing_box.to('cuda').unsqueeze(0), instance_mask)

        unique_class = torch.unique(recon_final_seg[torch.bitwise_not(bool_mask)], sorted=True)

        nothing_prompt = 'high resolution, natural images'
        inter_image = pipe(prompt=nothing_prompt, image=init_image, mask_image=inverted_mask_image).images[0]
        image_list = []
        prompt_list = []
        mask_list = []
        mask_list.append(inverted_mask_image)
        image_list.append(inter_image)
        prompt_list.append(nothing_prompt)

        for class_index in unique_class:
            mask = (recon_final_seg == class_index.item())*(torch.bitwise_not(bool_mask.to('cuda')).int()).int()*255
            inverted_mask_image = PIL.Image.fromarray(np.array(mask.cpu().squeeze()).astype(np.uint8)).convert("RGB")
            inter_mask = (recon_final_seg == class_index.item())*(torch.bitwise_not(bool_mask.to('cuda')).int()).int()*255
            class_name = index2clsname[class_index.item()]
            prompt = 'a ' + f'{class_name}'
            # print(prompt)
            inter_image = pipe(prompt=prompt, image=init_image, mask_image=inverted_mask_image).images[0]
            inter_image = Image.composite(inter_image, image_list[-1], inverted_mask_image.convert('L'))
            mask_list.append(inverted_mask_image)
            image_list.append(inter_image)
            prompt_list.append(prompt)


        image_list[0].save(os.path.join(args.output_path, 'init_images', f'{file_name}.png'))
        image_list[-1].save(os.path.join(args.output_path, 'images', f'{file_name}.png'))

def fix_seed(seed):
    if seed == -1:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
    else:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--coco_path", type=str, required=True, default="./coco", help="path to coco dataset")
    parser.add_argument("--modified_coco_path", type=str, required=True, default="./modified_coco", help="path to save modified coco dataset")
    parser.add_argument("--mit_ckpt_path", type=str, required=True, default="./ckpt", help="path to load checkpoints")
    parser.add_argument("--mgan_ckpt_path", type=str, required=True, default="./ckpt", help="path to load checkpoints")
    parser.add_argument("--bsc_ckpt_path", type=str, required=True, default="./ckpt", help="path to load checkpoints")
    parser.add_argument("--output_path", type=str, required=True, default="./output", help="path to save output images")
    parser.add_argument("--start", type=float, default=0.0, help="start index")
    parser.add_argument("--end", type=float, default=1.0, help="end index")
    args = parser.parse_args()
    main(args)