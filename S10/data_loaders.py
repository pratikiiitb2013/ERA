import torch
import torchvision
from data_aug import CustomTransforms


# dataloader arguments - something you'll fetch these from cmdprmt

def train_loader(batch_size, num_workers, cuda_available, means, stds, train=True, pin_memory=True):
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory) if cuda_available else dict(shuffle=True, batch_size=64)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=CustomTransforms(means, stds, train))
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    return trainloader


def test_loader(batch_size, num_workers, cuda_available, means, stds, train=False, pin_memory=True):
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory) if cuda_available else dict(shuffle=True, batch_size=64)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=CustomTransforms(means, stds, train))
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return testloader



