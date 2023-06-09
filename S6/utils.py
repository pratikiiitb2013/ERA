
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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

def download_and_get_data(train = True):
    return datasets.MNIST('../data', train=train, download=True, transform=get_transforms(train=train))


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def plt_graph(train_losses,test_losses,train_acc,test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")