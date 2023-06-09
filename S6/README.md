
# ERA - Session 6 Solution

### Packages and libraries

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-397/) [![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-green.svg)](https://pytorch.org/) [![torchvision 0.15+](https://img.shields.io/badge/torchvision-0.15+-blue.svg)](https://pypi.org/project/torchvision/) [![torch-summary 1.4](https://img.shields.io/badge/torchsummary-1.4+-green.svg)](https://pypi.org/project/torch-summary/)

Part 1 - Working Backpropagation on excel
------
#### Simple Neural net architecture for Excel

![alt text](https://github.com/pratikiiitb2013/ERA/blob/main/S6/images/Network_backprop.JPG)

#### Computation Snippet(Refer [attached excel](https://github.com/pratikiiitb2013/ERA/blob/main/S6/BackProp_PratikPractice.xlsx) for details)

![alt text](https://github.com/pratikiiitb2013/ERA/blob/main/S6/images/Calculations_backprop.JPG)


#### Major Steps:
  - Design the neural network as shown in above screenshot, with the given inputs, targets and initial weights.
  - Next major step is to calculate the h1,h2, and the activated h1 and h2 that is nothing but sigmoid of h1 and h2 respectively as we are using activation function as Sigmoid.
  - similarly we calulate o1,o2 and the coressponding activated values.
  - Now the next step is to understand that total Error is the combination of two components coming from two branches and is denoted by E1 and E2.
  - This E1 and E2 is the Mean square error, that is calulated as the square of the difference between expected output and actual output along the two branches. Here 1/2 is included just for simplicity of calulation while taking derivatives.
  - Following equations can be derived :
      - h1 = w1 * i1 + w2 * i2
      - h2 = w3 * i1 + w4 * i2
      - a_h1 = σ(h1) = 1/(1+exp(-h1))
      - a_h2 = σ(h2)
      - o1 = w5 * a_h1 + w6 * a_h2
      - o2 = w7 * a_h1 + w8 * a_h2
      - a_o1 = σ(o1)
      - a_o2 = σ(o2)
      - E1 = ½*(t1 - a_o1)²
      - E2 = ½*(t2 - a_o2)²
      - E_Total = E1 + E2
      
  - Now with backpropagation we move in backward direction, calculating the rate of change in total error with respect individual weights keeping other weights as constant.
  - We first calculate δE_t/δw5, here we see that w5 has impact only along the path E1 and by using chain rule we can rewrite the equation as : (δE1/δa_o1)*(δa_o1/δo1)*(δo1/δw5)
  - We then calculate each of the components of this equation separately. 
  - And finally we combine the parts together to get : δE_t/δw5 = (a_o1-t1)*(a_o1*(1-ao1))*a_h1
  - And carefully observing the paths for each of the following weights we can rewrite equation for them as :
        δE_t/δw6 = (a_o1-t1)*(a_o1*(1-ao1))*a_h2
        δE_t/δw7 = (a_o2-t2)*(a_o2*(1-ao2))*a_h1
        δE_t/δw8 = (a_o2-t2)*(a_o2*(1-ao2))*a_h2
  - Now moving further in backpropagation , we see the requirement to calculate similar rate of change of total error with respect to other weights.
  - For this we have to get δE_t/δw1 , before calculating it directly we need to get equation for δE_t/δa_h1
  - This a_h1 will have impact along both the paths, ie along the weight w5 and w7, so the impact of a_h1 is distributed along E1 and E2.
  - For calculating δE_t/δa_h1 , we have to combine the equation for δE1/δa_h1 and δE2/δa_h1
  - From the chain rule we can rewrite the equation as δE1/δa_h1 = (δE1/δa_o1)*(δa_o1/δo1)*(δo1/δa_h1)
  - Above one on simplification gives (a_o1 - t1)*(a_o1*(1-a_o1))*w5
  - From the chain rule we can rewrite the equation as δE2/δa_h1 = (δE2/δa_o2)*(δa_o2/δo2)*(δo2/a_h1)
  - Above one on simplification gives (a_o2-t2)*(a_o2*(1-a_o2))*w7
  - So, we get δE_t/δa_h1  = ((a_o1 - t1)*(a_o1*(1-a_o1))*w5) + ((a_o2-t2)*(a_o2*(1-a_o2))*w7)
  - Now to calculate δE_t/δw1 we can again use the chain rule and rewrite equation as δE_t/δw1 = (δE_t/δa_o1)*(δa_o1/δo1)*(δo1/δa_h1)*(δa_h1/δh1)*(δh1/δw1)
  - If we see carefully then the combination of first three components on RHS is nothing but δE_t/δa_h1, so we can rewrite the equation as :
         δE_t/δw1 = (δE_t/δa_h1)*(δa_h1/δh1)*(δh1/δw1) =  (δE_t/δa_h1)*(a_h1*(1-a_h1))*(i1)  , or by putting above calculated value for δE_t/δa_h1 in equation we get
         δE_t/δw1 =( ((a_o1 - t1)*(a_o1*(1-a_o1))*w5)+((a_o2-t2)*(a_o2*(1-a_o2))*w7)*(a_h1*(1-a_h1))*(i1)
  - Similarly we can derive equations by considering the path they affect :
         δE_t/δw2 = (δE_t/δa_h1)*(a_h1*(1-a_h1))*(i2)
         δE_t/δw3 = (δE_t/δa_h2)*(a_h2*(1-a_h2))*(i1)
         δE_t/δw4 = (δE_t/δa_h2)*(a_h2*(1-a_h2))*(i2)
  - For above equations of δE_t/δw3 and δE_t/δw4 we need calculation of δE_t/δa_h2
  - Again we can simplify δE_t/δa_h2 as δ(E1+E2)/δa_h2
  - Using the chain rule we can rewrite above two terms as :
         δE1/δa_h2 = (δE1/δa_o1)*(δa_o1/δo1)*(δo1/δa_h2), in this we have already calculated δE1/δa_o1 and δa_o1/δo1 previously for δE1/δa_h1
         δE2/δa_h2 = (δE2/δa_o2)*(δa_o2/δo2)*(δo2/δa_h2), in this we have already calculated δE2/δa_o2 and δa_o2/δo2 previously for δE2/δa_h1
  - so we can rewrite above equations as :
         δE1/δa_h2 =  (a_o1-t1)*(a_o1*(1-a_o1))*w6
         δE2/δa_h2 =  (a_o2-t2)*(a_o2*(1-a_o2))*w8
  - Combining both we get , 
         δE_t/δa_h2 = ((a_o1-t1)*(a_o1*(1-a_o1))*w6) + ((a_o2-t2)*(a_o2*(1-a_o2))*w8)
  - Now when creating the table given inputs, targets and initial weights are populated, calculated value of h1, h2, a_h1, a_h2, o1, o2, a_o1, a_o2, E1, E2, E_Total, δE_t/δw1,	δE_t/δw2,	δE_t/δw3,	δE_t/δw4,	δE_t/δw5,	δE_t/δw6,	δE_t/δw7, and	δE_t/δw8.
  - Then we calculate the weights for next phase, going in backpropagation style and starting from w5, the new w5(new) = w5(old) - η*(δE_t/δw5), here η is learning rate.
  - Then we calculate the new weights for others as well in same way, we calculate w6(new), w7(new), w8(new) and in next step we calculate w1(new), w2(new), w3(new) and w4(new).
  - We then extend this table till a significant number of levels so that we could observe that E_Total is decreasing with each level and activated outputs a_o1 and a_o2 move more closure to the expected target t1 and t2.
  - Select the column for E_Total and create a graph to observe the pattern of decreasing E_Total.
 
 <i>Note: We have used derivative of sigmoid function  as :
  {1/(1+exp(-x))}*{1 - (1/(1+exp(-x)))} = σ(x)*(1-σ(x))</i>
 
 
 
#### Variation in Error as we increase LR [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
![alt text](https://github.com/pratikiiitb2013/ERA/blob/main/S6/images/different_LR_backprop.JPG)


Part 2 - Better model with <20k paramemters
------
We have used the modular codes as earlier. Out main code runs in [ERAS6_pratik.ipynb](https://github.com/pratikiiitb2013/ERA/blob/main/S6/ERAS6_pratik.ipynb) file where we will be calling classes and functions from other 2 files, [utils.py](https://github.com/pratikiiitb2013/ERA/blob/main/S6/utils.py) and [model.py](https://github.com/pratikiiitb2013/ERA/blob/main/S6/model.py)

#### Model Code
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) #input 28x28x3 OUtput 26x26x16 RF 3
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3) #input 26x26x16 OUtput 24x24x16 RF 5
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, 3) #input 24x24x16 OUtput 22x22x16 RF 7
        self.bn3 = nn.BatchNorm2d(16)
        
        self.pool1 = nn.MaxPool2d(2, 2)  #input 22x22x16 OUtput 11x11x16 RF 8
  
     

        self.conv4 = nn.Conv2d(16, 16, 1) #input 11x11x16 OUtput 11x11x8 RF 8
        self.bn4 = nn.BatchNorm2d(16)
        
        self.conv5 = nn.Conv2d(16, 16, 3) #input 11x11x8 OUtput 9x9x16 RF 12
        self.bn5 = nn.BatchNorm2d(16)
        self.dp5 = nn.Dropout2d(0.1)

        self.conv6 = nn.Conv2d(16, 16, 3) #input 9x9x16 OUtput 7x7x16 RF 16
        self.bn6 = nn.BatchNorm2d(16)
        self.dp6 = nn.Dropout2d(0.1)

        self.conv7 = nn.Conv2d(16, 16, 3) #input 7x7x16 OUtput 5x5x16 RF 20
        self.bn7 = nn.BatchNorm2d(16)
        self.dp7 = nn.Dropout2d(0.1)

        self.conv8 = nn.Conv2d(16, 16, 3, padding=1) #input 5x5x16 OUtput 1x1x10 RF 2
        self.bn8 = nn.BatchNorm2d(16)
        self.dp8 = nn.Dropout2d(0.2)

        self.conv9 = nn.Conv2d(16, 16, 3, padding=1) #input 5x5x16 OUtput 1x1x10 RF 2
        self.bn9 = nn.BatchNorm2d(16)
        self.dp9 = nn.Dropout2d(0.2)
        
        self.gap = nn.AvgPool2d(kernel_size=5)
        self.linear = nn.Linear(16,10)

    def forward(self, x):
        x = (self.bn1(F.relu(self.conv1(x)))) 
        x = (self.bn2(F.relu(self.conv2(x)))) 
        x = self.pool1((self.bn3(F.relu(self.conv3(x))))) 
        x = self.bn4(F.relu(self.conv4(x))) 
        x = self.dp5(self.bn5(F.relu(self.conv5(x)))) 
        x = self.dp6(self.bn6(F.relu(self.conv6(x)))) 
        x = self.dp7(self.bn7(F.relu(self.conv7(x)))) 
        x = self.dp8(self.bn8(F.relu(self.conv8(x)))) 
        x = self.dp9(self.bn9(F.relu(self.conv9(x)))) 
        x = self.gap(x)
        #x = self.conv8(x) 
        x = x.view(-1, 16)
        x = self.linear(x)
        return F.log_softmax(x)
```

#### Model summary
We have used a simple model with following architecture with <b>17,130</b> parameters.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
       BatchNorm2d-2           [-1, 16, 26, 26]              32
            Conv2d-3           [-1, 16, 24, 24]           2,320
       BatchNorm2d-4           [-1, 16, 24, 24]              32
            Conv2d-5           [-1, 16, 22, 22]           2,320
       BatchNorm2d-6           [-1, 16, 22, 22]              32
         MaxPool2d-7           [-1, 16, 11, 11]               0
            Conv2d-8           [-1, 16, 11, 11]             272
       BatchNorm2d-9           [-1, 16, 11, 11]              32
           Conv2d-10             [-1, 16, 9, 9]           2,320
      BatchNorm2d-11             [-1, 16, 9, 9]              32
        Dropout2d-12             [-1, 16, 9, 9]               0
           Conv2d-13             [-1, 16, 7, 7]           2,320
      BatchNorm2d-14             [-1, 16, 7, 7]              32
        Dropout2d-15             [-1, 16, 7, 7]               0
           Conv2d-16             [-1, 16, 5, 5]           2,320
      BatchNorm2d-17             [-1, 16, 5, 5]              32
        Dropout2d-18             [-1, 16, 5, 5]               0
           Conv2d-19             [-1, 16, 5, 5]           2,320
      BatchNorm2d-20             [-1, 16, 5, 5]              32
        Dropout2d-21             [-1, 16, 5, 5]               0
           Conv2d-22             [-1, 16, 5, 5]           2,320
      BatchNorm2d-23             [-1, 16, 5, 5]              32
        Dropout2d-24             [-1, 16, 5, 5]               0
        AvgPool2d-25             [-1, 16, 1, 1]               0
           Linear-26                   [-1, 10]             170
================================================================
Total params: 17,130
Trainable params: 17,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.54
Params size (MB): 0.07
Estimated Total Size (MB): 0.61
```

Finally reached <b>99.51%</b> accuracy on test images in 20 epochs.
```
Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 17
Train: Loss=0.0151 Batch_id=468 Accuracy=98.20: 100%|██████████| 469/469 [00:54<00:00,  8.61it/s]
Test set: Average loss: 0.0001, Accuracy: 9950/10000 (99.50%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.1117 Batch_id=468 Accuracy=98.24: 100%|██████████| 469/469 [02:38<00:00,  2.95it/s]
Test set: Average loss: 0.0001, Accuracy: 9951/10000 (99.51%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.0971 Batch_id=468 Accuracy=98.32: 100%|██████████| 469/469 [01:28<00:00,  5.32it/s]
Test set: Average loss: 0.0002, Accuracy: 9948/10000 (99.48%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0445 Batch_id=468 Accuracy=98.33: 100%|██████████| 469/469 [03:01<00:00,  2.58it/s]
Test set: Average loss: 0.0001, Accuracy: 9951/10000 (99.51%)

Adjusting learning rate of group 0 to 1.0000e-03.
```
---
# References
Special thanks to [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/) for his guidance.
Do checkout his AI courses on [THE SCHOOL OF AI](https://theschoolof.ai/)

