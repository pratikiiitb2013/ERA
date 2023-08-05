import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Trainval import traine, teste

def define_optim_criterion(net):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=1e-4)
  return criterion, optimizer

def run_train_eval(scheduler, optimizer, net, device, trainloader, testloader, criterion):
 test_acc = []
 train_acc = []
 learning_rate = []
 for epoch in range(20):  # loop over the dataset multiple times
    scheduler.step()
    for param_group in optimizer.param_groups:
      print("lr= ",param_group['lr'])
    train_acc.append(traine.train(net, device, trainloader, optimizer, criterion, epoch))
    test_acc.append(teste.test(net, device, testloader))
    learning_rate.append(param_group['lr'])
  
 print('Finished Training')
 return train_acc, test_acc, learning_rate