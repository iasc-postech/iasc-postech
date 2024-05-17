from typing import Any, Callable, List, Optional, Union, Dict
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


import clip
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

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

