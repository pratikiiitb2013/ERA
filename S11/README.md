# ERA - Session 9 Solution

### Packages and libraries

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-397/) [![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-green.svg)](https://pytorch.org/) [![torchvision 0.15+](https://img.shields.io/badge/torchvision-0.15+-blue.svg)](https://pypi.org/project/torchvision/) [![torch-summary 1.4](https://img.shields.io/badge/torchsummary-1.4+-green.svg)](https://pypi.org/project/torch-summary/)

Assignment Details
------
![Assignment](https://github.com/pratikiiitb2013/ERA/blob/main/S11/images/assignment.JPG)

Model Architecture
------

As shown in below image, Resnet18 architecture is used. Refer [model code](https://github.com/pratikiiitb2013/ERA/blob/main/S11/models/resnet.py) for details


<img src="https://github.com/pratikiiitb2013/ERA/blob/main/S11/images/model_architecture.png" alt="drawing" width="35%" height="35%"/>


Image Augmentation Impelmetation Examples
-------
![alt text](https://github.com/pratikiiitb2013/ERA/blob/main/S11/images/augmented_images_sample.png)

#### Model summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
```

Lr finder curve to determine max_lr
------
As shown below, with the help of LR finder a Max LR of <b>5.22E-02</b> is choosen to train the model through one-cycle LR policy
![alt text](https://github.com/pratikiiitb2013/ERA/blob/main/S11/images/LR_finder.png)

#### Training Logs
Reached <b>91.63%</b> accuracy on test images in 20 epochs.
```
lr=  0.000522
Epoch=1 Loss=1.6114585399627686 Batch_id=97 Accuracy=34.70: 100%|██████████| 98/98 [00:44<00:00,  2.20it/s]

Test set: Average loss: 1.7408, Accuracy: 4183/10000 (41.83%)

lr=  0.009149672913117546
Epoch=2 Loss=1.4076814651489258 Batch_id=97 Accuracy=45.40: 100%|██████████| 98/98 [00:43<00:00,  2.26it/s]

Test set: Average loss: 1.4851, Accuracy: 4881/10000 (48.81%)

lr=  0.017777345826235094
Epoch=3 Loss=1.4079077243804932 Batch_id=97 Accuracy=52.75: 100%|██████████| 98/98 [00:43<00:00,  2.27it/s]

Test set: Average loss: 1.8146, Accuracy: 4477/10000 (44.77%)

lr=  0.02640501873935264
Epoch=4 Loss=0.9798606038093567 Batch_id=97 Accuracy=58.11: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 1.5923, Accuracy: 5147/10000 (51.47%)

lr=  0.03503269165247019
Epoch=5 Loss=1.0451549291610718 Batch_id=97 Accuracy=61.29: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 1.8167, Accuracy: 5058/10000 (50.58%)

lr=  0.043660364565587736
Epoch=6 Loss=1.0466821193695068 Batch_id=97 Accuracy=64.43: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 1.2101, Accuracy: 6299/10000 (62.99%)

lr=  0.05216195715743441
Epoch=7 Loss=0.9099739789962769 Batch_id=97 Accuracy=67.83: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 0.8796, Accuracy: 7004/10000 (70.04%)

lr=  0.048433758586005835
Epoch=8 Loss=0.7489835619926453 Batch_id=97 Accuracy=71.61: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]

Test set: Average loss: 0.7332, Accuracy: 7608/10000 (76.08%)

lr=  0.04470556001457726
Epoch=9 Loss=0.8080465197563171 Batch_id=97 Accuracy=73.88: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 0.7060, Accuracy: 7589/10000 (75.89%)

lr=  0.04097736144314869
Epoch=10 Loss=0.6890819072723389 Batch_id=97 Accuracy=75.43: 100%|██████████| 98/98 [00:42<00:00,  2.28it/s]

Test set: Average loss: 0.6355, Accuracy: 7901/10000 (79.01%)

lr=  0.037249162871720115
Epoch=11 Loss=0.6856144666671753 Batch_id=97 Accuracy=77.61: 100%|██████████| 98/98 [00:43<00:00,  2.28it/s]

Test set: Average loss: 0.6149, Accuracy: 7893/10000 (78.93%)

lr=  0.03352096430029155
Epoch=12 Loss=0.6198899149894714 Batch_id=97 Accuracy=79.09: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 0.4980, Accuracy: 8301/10000 (83.01%)

lr=  0.029792765728862976
Epoch=13 Loss=0.5165726542472839 Batch_id=97 Accuracy=80.31: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]

Test set: Average loss: 0.5039, Accuracy: 8353/10000 (83.53%)

lr=  0.026064567157434403
Epoch=14 Loss=0.4894197881221771 Batch_id=97 Accuracy=81.57: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 0.4118, Accuracy: 8620/10000 (86.20%)

lr=  0.022336368586005833
Epoch=15 Loss=0.5067570805549622 Batch_id=97 Accuracy=82.95: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 0.3453, Accuracy: 8846/10000 (88.46%)

lr=  0.018608170014577263
Epoch=16 Loss=0.47367995977401733 Batch_id=97 Accuracy=83.92: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 0.3249, Accuracy: 8897/10000 (88.97%)

lr=  0.01487997144314869
Epoch=17 Loss=0.39824503660202026 Batch_id=97 Accuracy=84.87: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 0.3215, Accuracy: 8921/10000 (89.21%)

lr=  0.011151772871720117
Epoch=18 Loss=0.3808738589286804 Batch_id=97 Accuracy=86.08: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]

Test set: Average loss: 0.2878, Accuracy: 9040/10000 (90.40%)

lr=  0.007423574300291544
Epoch=19 Loss=0.3697373867034912 Batch_id=97 Accuracy=87.05: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]

Test set: Average loss: 0.2599, Accuracy: 9133/10000 (91.33%)

lr=  0.0036953757288629777
Epoch=20 Loss=0.3735744059085846 Batch_id=97 Accuracy=88.09: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]

Test set: Average loss: 0.2487, Accuracy: 9163/10000 (91.63%)
```

#### Wrong classified images examples and corresponding GradCam maps
<img src="https://github.com/pratikiiitb2013/ERA/blob/main/S11/images/incorrect_prediction.png" alt="drawing" width="50%" height="50%"/>
<br>
<img src="https://github.com/pratikiiitb2013/ERA/blob/main/S11/images/incorrect_prediction_gradcam.png" alt="drawing" width="50%" height="50%"/>


#### Correct classified images examples and corresponding GradCam 
<img src="https://github.com/pratikiiitb2013/ERA/blob/main/S11/images/correct_prediction.png" alt="drawing" width="50%" height="50%"/>
<br>
<img src="https://github.com/pratikiiitb2013/ERA/blob/main/S11/images/correct_prediction_gradcam.png" alt="drawing" width="50%" height="50%"/>


# References
Special thanks to [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/) for his guidance.
Do checkout his AI courses on [THE SCHOOL OF AI](https://theschoolof.ai/)
