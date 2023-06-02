
from torchvision import datasets, transforms

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

