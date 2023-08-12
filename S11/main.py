import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loaders import train_loader, test_loader
# from utils_visualization import visualize_data_loader, wrong_and_correct_classified_data, plot_with_actual_predicted_class, get_random_index
from utils_visualization import visualize_data_samples, wrong_and_correct_classified_data, plot_with_actual_predicted_class, get_random_index, display_gradcam_output
# from custom_resnet import Net
from models.resnet import *
from train_test import train, test



import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


########################################################

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

means=[0.485,0.456,0.406]
stds=[0.229,0.224,0.225]

inv_means=[-0.485/0.229, -0.456/0.224, -0.406/0.225]
inv_stds=[1/0.229, 1/0.224, 1/0.225]

batch_size = 512
num_workers = 2
cuda_available = torch.cuda.is_available()
print(cuda_available)

################################################################

trainldr = train_loader(batch_size, num_workers, cuda_available, means, stds)
testldr = test_loader(batch_size, num_workers, cuda_available, means, stds)

###############################################################

dataiter = iter(trainldr)
image_tensors, labels = next(dataiter)
print(image_tensors.shape, type(image_tensors))

number_of_samples = 25

visualize_data_samples(image_tensors, labels, classes, number_of_samples, inv_means, inv_stds)
######################################################################################

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
net = ResNet18().to(device)
print(summary(net, input_size=(3, 32, 32)))
######################################################################################

import torch.nn as nn
from torch_lr_finder import LRFinder
train_criterion = nn.CrossEntropyLoss()
test_criterion = nn.CrossEntropyLoss(reduction='sum')
# train_criterion = F.nll_loss
# test_criterion = F.nll_loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=1e-4)
lr_finder = LRFinder(net, optimizer, train_criterion, device='cuda')
lr_finder.range_test(trainldr,end_lr=10, num_iter=200, step_mode='exp')
lr_finder.plot()
lr_finder.reset()

##################################################################################################


import torch
from tqdm import tqdm
from torch.optim import lr_scheduler

EPOCHS = 20

model =  ResNet18().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = lr_scheduler.OneCycleLR(optimizer,
                                    max_lr=5.22E-02,
                                    steps_per_epoch=len(trainldr),
                                    epochs=EPOCHS,
                                    # pct_start=5/EPOCHS,
                                    pct_start=int(0.3*EPOCHS)/EPOCHS if EPOCHS != 1 else 0.5,   # 30% of total number of Epochs
                                    div_factor=100,
                                    final_div_factor=100,
                                    three_phase=False,
                                    anneal_strategy='linear')


train_losses = []
test_losses = []
train_acc = []
test_acc = []



# learning_rate = []
# EPOCHS = 20
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    for param_group in optimizer.param_groups:
      print("lr= ",param_group['lr'])
    train_losses, train_acc = train(model, device, trainldr, optimizer, epoch, scheduler, train_criterion, train_losses, train_acc)
    test_loss, test_acc = test(model, device, testldr,test_criterion, test_losses, test_acc)
print('Finished Training')


#################################################################################################################

correct_preds,correct_target,correct_img_data, incorrect_preds,incorrect_target,incorrect_img_data = wrong_and_correct_classified_data(model, testldr, device, classes)

correct_idxs = get_random_index(correct_img_data)
incorrect_idxs = get_random_index(incorrect_img_data)

##########################################################################################################

trgt, imgs, prds = plot_with_actual_predicted_class(correct_target, correct_img_data, correct_preds,correct_idxs, classes,number_of_samples, inv_means,inv_stds)
target_layers = [model.layer4[-1]]
targets = None
display_gradcam_output(imgs, trgt.squeeze(), prds.squeeze(), classes, model, inv_means, inv_stds, target_layers, targets, number_of_samples, transparency=0.70)


####################################################################################################################################

trgt, imgs, prds = plot_with_actual_predicted_class(incorrect_target, incorrect_img_data, incorrect_preds,incorrect_idxs,classes,number_of_samples, inv_means,inv_stds)
target_layers = [model.layer4[-1]]
targets = None
display_gradcam_output(imgs, trgt.squeeze(), prds.squeeze(), classes, model, inv_means, inv_stds, target_layers, targets, number_of_samples, transparency=0.70)

######################################################################################################################################

