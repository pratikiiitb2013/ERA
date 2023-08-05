import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch
from torch_lr_finder import LRFinder
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")

def getdevice():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device

def run_lrfinder(net, optimizer, criterion, device, trainloader):
  lr_finder = LRFinder(net, optimizer, criterion, device='cuda')
  lr_finder.range_test(trainloader,end_lr=10, num_iter=200, step_mode='exp')
  lr_finder.plot()
  lr_finder.reset()

def displayimage(trainloader, classes):
 # get some random training images
 dataiter = iter(trainloader)
 images, labels = next(dataiter)
 # print labels
 plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
 for i in range(0,25):
  plt.subplot(5, 5, i+1)
  frame1 = plt.gca()
  frame1.axes.xaxis.set_ticklabels([])
  frame1.axes.yaxis.set_ticklabels([])
  plt.imshow(np.transpose(((images[i]/2)+0.5).numpy(),(1,2,0)))
  plt.title(classes[labels[i]])

def plot_curves(train_acc, test_acc, learning_rate):
  fig, axs = plt.subplots(3)
  axs[0].plot(train_acc)
  axs[0].set_title("Training Accuracy")
  axs[0].set_xlabel("Batch")
  axs[0].set_ylabel("Accuracy")
  axs[1].plot(test_acc)
  axs[1].set_title("Test Accuracy")
  axs[1].set_xlabel("Batch")
  axs[2].plot(learning_rate)
  axs[2].set_title("Learning rate")
  axs[2].set_xlabel("epoch")
  axs[2].set_ylabel("lr")

def show_images(net, testloader, device, classes, flag):
  net.eval()
  missed = []
  pred = []
  targ = []
  empty_tensor = torch.tensor([]).to(device)
  with torch.no_grad():
      pbar1 = tqdm(testloader)
      for i, (data, target) in enumerate(pbar1):
           data, target = data.to(device), target.to(device)
           outputs = net(data)
           _, predicted = torch.max(outputs.data, 1)
           target1 = target.cpu().numpy()
           predicted1 = predicted.cpu().numpy()
           for i in range(64):
             if flag==1:
              if target1[i]==predicted1[i]:
                 missed.append(i)
                 new_tensor = data[i].unsqueeze(0)
                 empty_tensor = torch.cat((empty_tensor, new_tensor), dim=0)
                 pred.append(predicted1[i])
                 targ.append(target1[i])
             else:
              if target1[i]!=predicted1[i]:
                 missed.append(i)
                 new_tensor = data[i].unsqueeze(0)
                 empty_tensor = torch.cat((empty_tensor, new_tensor), dim=0)
                 pred.append(predicted1[i])
                 targ.append(target1[i])
           break

  plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
  for i in range(0,10):
   plt.subplot(5, 2, i+1)
   frame1 = plt.gca()
   frame1.axes.xaxis.set_ticklabels([])
   frame1.axes.yaxis.set_ticklabels([])
   plt.imshow(np.transpose(((data[missed[i]].cpu()/2)+0.5).numpy(),(1,2,0)))
   plt.ylabel("GT:"+str(classes[target1[missed[i]]])+'\nPred:'+str(classes[predicted1[missed[i]]]))
  return empty_tensor, pred, targ

def gradcame(net, img, targ, image_tensor):
  target_layers = [net.layer4[-1]]
  # Construct the CAM object once, and then re-use it on many images:
  cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
  # We have to specify the target we want to generate
  # the Class Activation Maps for.
  # If targets is None, the highest scoring category
  # will be used for every image in the batch.
  # Here we use ClassifierOutputTarget, but you can define your own custom targets
  # That are, for example, combinations of categories, or specific outputs in a non standard model.
  targets = [ClassifierOutputTarget(targ[img])]

  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
  grayscale_cam = cam(input_tensor=image_tensor[img].unsqueeze(0), targets=targets)

  # In this example grayscale_cam has only one image in the batch:
  grayscale_cam = grayscale_cam[0, :]

  input_image = image_tensor[img].permute(1, 2, 0)
  input_image_normalized = (input_image - input_image.min()) / (input_image.max() - input_image.min())

  # Convert the normalized image tensor to NumPy array and change data type to np.float32
  input_image_np = input_image_normalized.cpu().numpy().astype(np.float32)

  visualization = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)
  return input_image_np, visualization

def visualize_gradcam(net, image_tensor, targ, pred, classes):
  plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.2)
  maxi = 10
  # Set the figure size to adjust the size of the displayed images
  plt.figure(figsize=(10, 2 * maxi))  # Adjust the width and height as needed
  for img in range(maxi):
    input_image_np,visualization=gradcame(net, img, targ, image_tensor)
    plt.subplot(maxi,2,2*img+1)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.imshow(input_image_np)
    plt.subplot(maxi,2,2*img+2)
    plt.ylabel("GT:"+str(classes[targ[img]])+'\nPred:'+str(classes[pred[img]]))
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.imshow(visualization)
