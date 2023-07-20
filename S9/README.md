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

Image AUgmentation Impelmetation Exapmles
-------
![alt text](https://github.com/pratikiiitb2013/ERA/blob/main/S9/images/augmented_images.png)
