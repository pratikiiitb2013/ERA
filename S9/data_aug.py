from albumentations import *
from albumentations.pytorch import ToTensorV2

from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from albumentations.augmentations.geometric.transforms import PadIfNeeded
from albumentations.augmentations.crops.transforms import CenterCrop

import numpy as np


class TrainAlbumentation():
  def __init__(self):
    self.train_transform = Compose([
        Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
      ),
      HorizontalFlip(),
      ShiftScaleRotate(shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2), rotate_limit=(-15, 15), p=0.5),
      PadIfNeeded(min_height=48, min_width=48, pad_height_divisor=None, pad_width_divisor=None, p=1.0),
      CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[0.485,0.456,0.406], mask_fill_value = None),
      CenterCrop(32, 32, always_apply=False, p=1.0),

      ToTensorV2()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.train_transform(image = img)['image']
    return img


class TestAlbumentation():
  def __init__(self):
    self.test_transform = Compose(
    [
      Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
      ),
      ToTensorV2()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.test_transform(image = img)['image']
    return img