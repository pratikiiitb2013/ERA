
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
 
 <b>Note: We have used derivative of sigmoid function  as :
  {1/(1+exp(-x))}*{1 - (1/(1+exp(-x)))} = σ(x)*(1-σ(x))</b>
  
#### Variation in Error as we increaseLR [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]

![alt text](https://github.com/pratikiiitb2013/ERA/blob/main/S6/images/different_LR_backprop.JPG)
