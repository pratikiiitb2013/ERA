# ERA - Session 9 Solution

### Packages and libraries

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-397/) [![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-green.svg)](https://pytorch.org/) [![torchvision 0.15+](https://img.shields.io/badge/torchvision-0.15+-blue.svg)](https://pypi.org/project/torchvision/) [![torch-summary 1.4](https://img.shields.io/badge/torchsummary-1.4+-green.svg)](https://pypi.org/project/torch-summary/)

Assignment Details
------
Write a new network that
- has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels        here instead of MP or strided convolution, then 200pts extra!)
- total RF must be more than 44
- one of the layers must use Depthwise Separable Convolution
- one of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- use albumentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

Model Architecture
------
As shown in below image, following architecture is followed
- 4 blocks with one initial conv -> conv0-> CB1-> CB2-> CB3-> OB
- Each of CB1, CB2 and CB3 contains
  - normal conv + dilated conv with padding
  - depthwise conv
  - transition layer -> 1X1 followed by dilated conv without padding(instead of MP)
- <b>Also, added input of each block CB1, CB2, CB3 to output of respective block through a skip connection 1X1 layer using stride of 2</b>
- OB has  normal conv + dilated conv with padding -> depthwise -> GAP -> (1X1 to #classes)
<img src="https://github.com/pratikiiitb2013/ERA/blob/main/S9/images/model_architechture.png" alt="drawing" width="50%" height="50%"/>

Image AUgmentation Impelmetation Examples
-------
![alt text](https://github.com/pratikiiitb2013/ERA/blob/main/S9/images/augmented_images.png)

#### Model summary(Refer [model code](https://github.com/pratikiiitb2013/ERA/blob/main/S9/model.py) for details)
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
       BatchNorm2d-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
       BatchNorm2d-6           [-1, 32, 32, 32]              64
              ReLU-7           [-1, 32, 32, 32]               0
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]           4,608
      BatchNorm2d-10           [-1, 32, 32, 32]              64
             ReLU-11           [-1, 32, 32, 32]               0
          Dropout-12           [-1, 32, 32, 32]               0
           Conv2d-13           [-1, 32, 32, 32]             320
           Conv2d-14          [-1, 128, 32, 32]           4,224
depthwise_separable_conv-15          [-1, 128, 32, 32]               0
      BatchNorm2d-16          [-1, 128, 32, 32]             256
             ReLU-17          [-1, 128, 32, 32]               0
          Dropout-18          [-1, 128, 32, 32]               0
           Conv2d-19           [-1, 32, 32, 32]           4,096
           Conv2d-20           [-1, 32, 16, 16]           9,216
           Conv2d-21           [-1, 32, 16, 16]             512
           Conv2d-22           [-1, 64, 16, 16]          18,432
      BatchNorm2d-23           [-1, 64, 16, 16]             128
             ReLU-24           [-1, 64, 16, 16]               0
          Dropout-25           [-1, 64, 16, 16]               0
           Conv2d-26           [-1, 64, 16, 16]          18,432
      BatchNorm2d-27           [-1, 64, 16, 16]             128
             ReLU-28           [-1, 64, 16, 16]               0
          Dropout-29           [-1, 64, 16, 16]               0
           Conv2d-30           [-1, 64, 16, 16]             640
           Conv2d-31          [-1, 128, 16, 16]           8,320
depthwise_separable_conv-32          [-1, 128, 16, 16]               0
      BatchNorm2d-33          [-1, 128, 16, 16]             256
             ReLU-34          [-1, 128, 16, 16]               0
          Dropout-35          [-1, 128, 16, 16]               0
           Conv2d-36           [-1, 32, 16, 16]           4,096
           Conv2d-37             [-1, 32, 8, 8]           9,216
           Conv2d-38             [-1, 32, 8, 8]           1,024
           Conv2d-39             [-1, 64, 8, 8]          18,432
      BatchNorm2d-40             [-1, 64, 8, 8]             128
             ReLU-41             [-1, 64, 8, 8]               0
          Dropout-42             [-1, 64, 8, 8]               0
           Conv2d-43             [-1, 64, 8, 8]          18,432
      BatchNorm2d-44             [-1, 64, 8, 8]             128
             ReLU-45             [-1, 64, 8, 8]               0
          Dropout-46             [-1, 64, 8, 8]               0
           Conv2d-47             [-1, 64, 8, 8]             640
           Conv2d-48            [-1, 128, 8, 8]           8,320
depthwise_separable_conv-49            [-1, 128, 8, 8]               0
      BatchNorm2d-50            [-1, 128, 8, 8]             256
             ReLU-51            [-1, 128, 8, 8]               0
          Dropout-52            [-1, 128, 8, 8]               0
           Conv2d-53             [-1, 32, 8, 8]           4,096
           Conv2d-54             [-1, 32, 4, 4]           9,216
           Conv2d-55             [-1, 32, 4, 4]           1,024
           Conv2d-56             [-1, 64, 4, 4]          18,432
      BatchNorm2d-57             [-1, 64, 4, 4]             128
             ReLU-58             [-1, 64, 4, 4]               0
          Dropout-59             [-1, 64, 4, 4]               0
           Conv2d-60             [-1, 64, 4, 4]          18,432
      BatchNorm2d-61             [-1, 64, 4, 4]             128
             ReLU-62             [-1, 64, 4, 4]               0
          Dropout-63             [-1, 64, 4, 4]               0
           Conv2d-64             [-1, 64, 4, 4]             640
           Conv2d-65            [-1, 128, 4, 4]           8,320
depthwise_separable_conv-66            [-1, 128, 4, 4]               0
      BatchNorm2d-67            [-1, 128, 4, 4]             256
             ReLU-68            [-1, 128, 4, 4]               0
          Dropout-69            [-1, 128, 4, 4]               0
        AvgPool2d-70            [-1, 128, 1, 1]               0
           Conv2d-71             [-1, 10, 1, 1]           1,280
================================================================
Total params: 197,392
Trainable params: 197,392
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.36
Params size (MB): 0.75
Estimated Total Size (MB): 12.13
----------------------------------------------------------------
```
#### Training Logs
Reached <b>85.7%</b> accuracy on test images in 100 epochs.
```
EPOCH: 95
Loss=0.5429221987724304 Batch_id=390 Accuracy=81.29: 100%|██████████| 391/391 [00:34<00:00, 11.42it/s]

Test set: Average loss: 0.4401, Accuracy: 8571/10000 (85.71%)

EPOCH: 96
Loss=0.4267800748348236 Batch_id=390 Accuracy=81.36: 100%|██████████| 391/391 [00:33<00:00, 11.79it/s]

Test set: Average loss: 0.4419, Accuracy: 8525/10000 (85.25%)

EPOCH: 97
Loss=0.484732449054718 Batch_id=390 Accuracy=81.30: 100%|██████████| 391/391 [00:36<00:00, 10.68it/s]

Test set: Average loss: 0.4782, Accuracy: 8466/10000 (84.66%)

EPOCH: 98
Loss=0.5143886208534241 Batch_id=390 Accuracy=81.32: 100%|██████████| 391/391 [00:35<00:00, 10.96it/s]

Test set: Average loss: 0.4512, Accuracy: 8523/10000 (85.23%)

EPOCH: 99
Loss=0.5892620086669922 Batch_id=390 Accuracy=81.58: 100%|██████████| 391/391 [00:40<00:00,  9.76it/s]

Test set: Average loss: 0.4250, Accuracy: 8578/10000 (85.78%)
```

#### Train/Test accuracy/Loss plots
![alt text](https://github.com/pratikiiitb2013/ERA/blob/main/S9/images/graphs.png)
