from typing import Any, Callable, List, Optional, Union, Dict
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import clip
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

import sys
sys.path.append("./OFA")
from OFA.utils.eval_utils import eval_step

from fairseq import utils
from fairseq import checkpoint_utils


from torchvision import transforms
from PIL import Image



class ClipPreprocess(nn.Module):
    def __init__(self, res):
        super(ClipPreprocess, self).__init__()
        self.MEAN = 255 * torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.STD = 255 * torch.tensor([0.26862954, 0.26130258, 0.27577711])
        self.res = res
    def forward(self, x):
        x = torch.round((F.interpolate(x, size=self.res, align_corners=True, mode='bicubic')*127.5 + 128).clamp(0.0, 255.0))
        x = (x - self.MEAN[None, :, None, None].type_as(x)) / self.STD[None, :, None, None].type_as(x)
        return x 

class CLIPscore(Metric):
    image_embed: List[np.ndarray]
    text_embed: List[np.ndarray]
    def __init__(
        self,
        compute_on_cpu : bool = True,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable[[Tensor], List[Tensor]] = None,
    ) -> None:

        super().__init__(
            #compute_on_cpu = compute_on_cpu,
            # compute_on_step=compute_on_step,
            # dist_sync_on_step=dist_sync_on_step,
            # process_group=process_group,
            # dist_sync_fn=dist_sync_fn,
        )

        self.add_state("image_embed", default=[], dist_reduce_fx="cat")
        self.add_state("text_embed", default=[], dist_reduce_fx="cat")
        self.CLIP_Pretrained, _ = clip.load("ViT-B/32")
        self.CLIP_Preprocess = ClipPreprocess(res=224)

    def update(self, input, type: str) -> None: # type = Image or Text

        if type == 'Image':
            image = self.CLIP_Preprocess(input)
            image_feature = self.CLIP_Pretrained.encode_image(image)
            self.image_embed.append(image_feature)
        elif type == 'Text':
            text = clip.tokenize(input).to(self.device)
            text_feature = self.CLIP_Pretrained.encode_text(text)
            self.text_embed.append(text_feature)
        else:
            raise NotImplementedError

    def compute(self, w=2.5) -> Tensor:
        import warnings
        import sklearn.preprocessing
        from packaging import version


        image_embed = dim_zero_cat(self.image_embed).cpu().numpy()
        text_embed = dim_zero_cat(self.text_embed).cpu().numpy()

        #as of numpy 1.21, normalize doesn't work properly for float16
        if version.parse(np.__version__) < version.parse('1.21'):
            images = sklearn.preprocessing.normalize(image_embed, axis=1)
            candidates = sklearn.preprocessing.normalize(text_embed, axis=1)
        else:
            warnings.warn(
                'due to a numerical instability, new numpy normalization is slightly different than paper results. '
                'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
            images = image_embed / np.sqrt(np.sum(image_embed**2, axis=1, keepdims=True))
            candidates = text_embed / np.sqrt(np.sum(text_embed**2, axis=1, keepdims=True))

        per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
        return np.mean(per), per, candidates


class VisualGroundingAcc(Metric):
    detected_result: List[torch.Tensor]
    def __init__(
        self,
        #compute_on_cpu : bool = True,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable[[Tensor], List[Tensor]] = None,
    ) -> None:
        super().__init__(
            #compute_on_cpu = compute_on_cpu,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        
        # Load pretrained ckpt & config
        overrides={"bpe_dir":"./OFA/utils/BPE"}
        self.models, self.cfg, self.task = checkpoint_utils.load_model_ensemble_and_task(
                utils.split_paths('./checkpoints/refcocog_large_best.pt'),
                arg_overrides=overrides
            )

        # Move models to GPU
        for model in self.models:
            model.eval()

        self.cfg.common.seed = 7
        self.cfg.generation.beam = 5
        self.cfg.generation.min_len = 4
        self.cfg.generation.max_len_a = 0
        self.cfg.generation.max_len_b = 4
        self.cfg.generation.no_repeat_ngram_size = 3

        # Fix seed for stochastic decoding
        if self.cfg.common.seed is not None and not self.cfg.generation.no_seed_provided:
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        self.patch_resize_transform = transforms.Compose([
            transforms.Resize((self.cfg.task.patch_image_size, self.cfg.task.patch_image_size), interpolation=Image.BICUBIC),
        ])

        # Text preprocess
        self.bos_item = torch.LongTensor([self.task.src_dict.bos()])
        self.eos_item = torch.LongTensor([self.task.src_dict.eos()])
        self.pad_idx = self.task.src_dict.pad()
        self.pad_item = torch.LongTensor([self.pad_idx])

        self.add_state("detected_result", default=[], dist_reduce_fx=torch.cat)

    def encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line=self.task.bpe.encode(text.lower()),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([ s , torch.cat([self.pad_item for i in range(s.shape[0], 30)]) ] )
        return s

    def construct_sample(self, image: Tensor, sample: Dict):
        w_resize_ratio = torch.tensor(self.cfg.task.patch_image_size / sample['ori_w'])
        h_resize_ratio = torch.tensor(self.cfg.task.patch_image_size / sample['ori_h'])
        patch_image = self.patch_resize_transform(image)
        patch_mask = sample['patch_mask']
        src_text = torch.cat([self.encode_text('which region does the text "{}" describe?'.format(text), append_bos=True, append_eos=True).unsqueeze(0) \
                        for text in sample['blip2_caption']], dim=0).to(patch_image.get_device())
        # print(src_text)
        src_length = torch.LongTensor([s.ne(self.pad_idx).long().sum() for s in src_text]).to(patch_image.get_device())
        processed_sample = {
            "id" : np.repeat(np.array(['None']), patch_mask.shape[0]),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": sample['patch_mask'].squeeze(),
            },
            "w_resize_ratios": w_resize_ratio,
            "h_resize_ratios": h_resize_ratio,
            "region_coords": sample['region_coords']
        }
        return processed_sample

    def update(self, image, sample):
        processed_sample = self.construct_sample(image, sample)
        current_result, scores = eval_step(self.task, self.generator, self.models, processed_sample)
        self.detected_result.append(scores) ##Coordinate 체크하기       

    def compute(self):
        total_detected = torch.count_nonzero(dim_zero_cat(self.detected_result))
        total_count = dim_zero_cat(self.detected_result).shape[0]
        acc = (total_detected/total_count)*100
        return acc


class mIoU(Metric):
    intersection: Tensor
    union: Tensor
    target: Tensor

    def __init__(
        self,
        class_numb = 200,
        ignore_index = 255,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable[[Tensor], List[Tensor]] = None,
    ) -> None:

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("intersection", default=torch.zeros(class_numb), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(class_numb), dist_reduce_fx="sum")
        self.add_state("target", default=torch.zeros(class_numb), dist_reduce_fx="sum")

        self.class_numb = class_numb
        self.ignore_index = ignore_index

    def intersectionAndUnionGPU(self, output, target):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape


        output = output.view(-1)
        target = target.view(-1)

        ## reduce zero label to ignore index for MSCOCO
        target[target == 0] = self.ignore_index
        target = target - 1
        target[target == self.ignore_index-1] = self.ignore_index

        output[output == 0] = self.ignore_index
        output = output - 1
        output[output == self.ignore_index-1] = self.ignore_index


        ## ignore specific index
        mask = (target != self.ignore_index)
        output = output[mask]
        target = target[mask]

        intersection = output[output == target]
        area_intersection = torch.histc(intersection, bins=self.class_numb, min=0, max=self.class_numb-1)
        area_output = torch.histc(output, bins=self.class_numb, min=0, max=self.class_numb-1)
        area_target = torch.histc(target, bins=self.class_numb, min=0, max=self.class_numb-1)
        area_union = area_output + area_target - area_intersection

        return area_intersection, area_union, area_target

    def update(self, pred, target) -> None: # type = Image or Text
        intersection, union, target = self.intersectionAndUnionGPU(pred, target)
        self.intersection += intersection
        self.union += union
        self.target += target

    def compute(self) -> Tensor:
        iou_class = self.intersection / self.union
        mIoU = torch.mean(iou_class[self.target!=0])
        return mIoU

