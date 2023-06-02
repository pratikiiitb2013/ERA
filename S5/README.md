
# ERA - Session 5 Solution


### Packages and libraries

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-397/) [![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-green.svg)](https://pytorch.org/) [![torchvision 0.15+](https://img.shields.io/badge/torchvision-0.15+-blue.svg)](https://pypi.org/project/torchvision/) [![torch-summary 1.4](https://img.shields.io/badge/torchsummary-1.4+-green.svg)](https://pypi.org/project/torch-summary/)

### Files
In this assigment we have divided the single code into following modular files. Out main code runs in [ERAS5_pratik.ipynb](https://github.com/pratikiiitb2013/ERA/blob/main/S5/ERAS5_pratik.ipynb) file where we will be calling classes and functions from other 2 files, [utils.py](https://github.com/pratikiiitb2013/ERA/blob/main/S5/utils.py) and [model.py](https://github.com/pratikiiitb2013/ERA/blob/main/S5/model.py)
- [utils.py](https://github.com/pratikiiitb2013/ERA/blob/main/S5/utils.py)
- [model.py](https://github.com/pratikiiitb2013/ERA/blob/main/S5/model.py)
- [ERAS5_pratik.ipynb](https://github.com/pratikiiitb2013/ERA/blob/main/S5/ERAS5_pratik.ipynb)

### Details of module files
#### 1. utils
This file contains utility functions for data download, data transforms and calculating count of correct predictions.
```python
def download_and_get_data(train = True):
    return datasets.MNIST('../data', train=train, download=True, transform=get_transforms(train=train))
```
```python
def get_transforms(train = True):
    if train:
        return transforms.Compose([
                transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
                transforms.Resize((28, 28)),
                transforms.RandomRotation((-15., 15.), fill=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
    else:
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) ## change 1
            ])
```
```python
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
```
#### 2. model
This file contains model class and 2 functions for train and test.
```python
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3,bias=False)
        self.fc1 = nn.Linear(4096, 50,bias=False)
        self.fc2 = nn.Linear(50, 10,bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)),2) 
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2) 
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```
```python
def train(model, device, train_loader, optimizer):
    train_losses = []
    train_acc = []
    
    model.train()
    pbar = tqdm(train_loader)
    
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = F.nll_loss(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
```
```python
def test(model, device, test_loader):
    test_losses = []
    test_acc = []

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
#### 3. ERAS5_pratik
This is the main code where we will be calling modules to train and test the model.
We first import modules from other files.
```python
from utils import get_transforms
from utils import download_and_get_data
from model import Net, train, test
```
Then we create data transforms by using imported function
```python
train_transforms = get_transforms(train=True)
test_transforms = get_transforms(train=False)
```
Then we download the train and test data
```python
train_data = download_and_get_data(train=True)
test_data = download_and_get_data(train=False)
```
Finally, we create model object by calling importted class 'Net'. Define optimizer and scheduler. And call train and test funtions.
```python
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
num_epochs = 20

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  train(model, device, train_loader, optimizer)
  test(model, device, test_loader)
  scheduler.step()
```

### Outputs and results
We have used a simple model with following architecture with <b>592,660</b> parameters.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
            Conv2d-2           [-1, 64, 24, 24]          18,432
            Conv2d-3          [-1, 128, 10, 10]          73,728
            Conv2d-4            [-1, 256, 8, 8]         294,912
            Linear-5                   [-1, 50]         204,800
            Linear-6                   [-1, 10]             500
================================================================
Total params: 592,660
Trainable params: 592,660
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.93
----------------------------------------------------------------
```
Finally reached <b>99.34%</b> accuracy on test images
```
Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0047 Batch_id=117 Accuracy=99.17: 100%|██████████| 118/118 [03:09<00:00,  1.61s/it]
Test set: Average loss: 0.0204, Accuracy: 9934/10000 (99.34%)
```
