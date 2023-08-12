from typing import Any
from albumentations import *
from albumentations.pytorch import ToTensorV2

from torchvision import transforms

from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from albumentations.augmentations.geometric.transforms import PadIfNeeded
from albumentations.augmentations.crops.transforms import CenterCrop, RandomCrop

import numpy as np


class CustomTransforms:
    def __init__(self, means, stds, train=True):
        if train:
            self.transforms = Compose([
                Normalize(mean=means, std=stds, always_apply=True),
                HorizontalFlip(),
                ShiftScaleRotate(shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2), rotate_limit=(-15, 15), p=0.5),
                PadIfNeeded(min_height=48, min_width=48, pad_height_divisor=None, pad_width_divisor=None, p=1.0),
                CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=means, mask_fill_value = None),
                # CenterCrop(32, 32, always_apply=False, p=1.0),
                RandomCrop(32, 32, always_apply=False, p=1.0),

                ToTensorV2()
            ])
        else:
            self.transforms = Compose([
                Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2()
            ])

    def __call__(self, img):
        img = np.array(img)
        img = self.transforms(image = img)['image']
        return img


# class UnNormalize:
#     def __init__(self, inv_mean, inv_std):
#         self.transforms = Compose([
#            Normalize(mean=inv_mean, std=inv_std, always_apply=True) 
#         ])


#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         # for t, m, s in zip(tensor, self.mean, self.std):
#         #     t.mul_(s).add_(m)
#         #     # The normalize code -> t.sub_(m).div_(s)
#         # return tensor
#         img = np.array(tensor)
#         img = self.transforms(image = img)['image']
#         return img


def inv_normalize(inv_means, inv_stds):
    return transforms.Normalize(mean=inv_means,std=inv_stds)