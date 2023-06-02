
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

