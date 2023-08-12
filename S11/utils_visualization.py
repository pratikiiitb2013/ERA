import torch
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from data_aug import inv_normalize
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# def visualize_data_loader(images, labels, classes, grid_size):
#     plt.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.4,
#                     hspace=0.4)
#     for i in range(0,25):
#         plt.subplot(grid_size, grid_size, i+1)
#         frame1 = plt.gca()
#         frame1.axes.xaxis.set_ticklabels([])
#         frame1.axes.yaxis.set_ticklabels([])
#         plt.imshow(np.transpose(((images[i]/2)+0.5).numpy(),(1,2,0)))
#         plt.title(classes[labels[i]])

def visualize_data_samples(image_tensors, labels, classes, number_of_samples, inv_means, inv_stds):
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.ceil(number_of_samples / x_count)
    inv_n = inv_normalize(inv_means, inv_stds)
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        img = image_tensors[i]
        # img = input_tensor.squeeze(0).to('cpu')
        img = inv_n(img)
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.xticks([])
        plt.yticks([])

def get_random_index(np_array, n=25):
  return np.random.choice(np_array.shape[0], n, replace=False)

def wrong_and_correct_classified_data(model, testloader, device, classes):
    correct_preds = []
    correct_target = []
    correct_img_data = []
    incorrect_preds = []
    incorrect_target = []
    incorrect_img_data = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.reshape(pred,(len(pred),1))
            target = np.reshape(target,(len(pred),1))
            data = data.cpu().numpy()
            # print(pred.shape,target.shape,data.shape)
            for i in range(len(pred)):
                if(pred[i]!=target[i]):
                    incorrect_preds.append(pred[i])
                    incorrect_target.append(target[i])
                    incorrect_img_data.append(data[i])
                else:
                    correct_preds.append(pred[i])
                    correct_target.append(target[i])
                    correct_img_data.append(data[i])

    return np.array(correct_preds),np.array(correct_target),np.array(correct_img_data), np.array(incorrect_preds), np.array(incorrect_target), np.array(incorrect_img_data)  

def plot_with_actual_predicted_class(true,ima,pred,idxs,classes,number_of_samples, inv_means, inv_stds):
    # print('Classes in order Actual and Predicted')
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.ceil(number_of_samples / x_count)
    inv_n = inv_normalize(inv_means, inv_stds)
    my_iterator = iter(idxs)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        next_element = next(my_iterator, random.choice(idxs))
        image,trgt,prd = ima[next_element],true[next_element][0],pred[next_element][0]
        image = torch.from_numpy(image)
        trgt = int(trgt)
        t = classes[trgt]
        prd = int(prd)
        p = classes[prd]
        # f = 'A:'+t + ',' +'P:'+p
        f =  r"Correct: " + t + '\n' + 'Pred: ' + p
        image = inv_n(image)
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        plt.tight_layout()
        plt.imshow(image)
        plt.title(f)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return true[idxs], ima[idxs], pred[idxs]


# def plot_with_actual_predicted_class(true,ima,pred,idxs,classes,n_figures = 10):
#     # print('Classes in order Actual and Predicted')
#     n_row = int(n_figures/2)
#     my_iterator = iter(idxs)
#     fig,axes = plt.subplots(figsize=(14, 10), nrows = n_row, ncols=2)
#     for ax in axes.flatten():
#         next_element = next(my_iterator, random.choice(idxs))
#         # a = random.randint(0,len(true)-1)

#         image,correct,wrong = ima[next_element],true[next_element][0],pred[next_element][0]
#         image = torch.from_numpy(image)
#         correct = int(correct)
#         c = classes[correct]
#         wrong = int(wrong)
#         w = classes[wrong]
#         f = 'A:'+c + ',' +'P:'+w
#         # if inv_normalize !=None:
#         #     image = inv_normalize(image)
#         image = np.transpose(((image/2)+0.5).numpy(),(1,2,0))
#         # image = image.numpy().transpose(1,2,0)
#         im = ax.imshow(image)
#         ax.set_title(f)
#         ax.axis('off')
#     plt.show()
#     return true[idxs], ima[idxs], pred[idxs]

def display_gradcam_output(data, true, pred, classes, model, inv_means, inv_stds, target_layers, targets=None, number_of_samples = 10, transparency = 0.60):
    """
    Function to visualize GradCam output on the data
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param model: Model architecture
    :param target_layers: Layers on which GradCam should be executed
    :param targets: Classes to be focused on for GradCam
    :param number_of_samples: Number of images to print
    :param transparency: Weight of Normal image when mixed with activations
    """
    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)
    inv_n = inv_normalize(inv_means, inv_stds)
    
    # Create an object for GradCam
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    # targets = [ClassifierOutputTarget(targ[img])]

    # Iterate over number of specified images
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = data[i]
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # # Get back the original image
        img = input_tensor.squeeze(0).to('cpu')
        img = inv_n(img)
        # img = inv_normalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()

        # # Mix the activations on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)

        # # Display the images on the plot
        plt.tight_layout()
        plt.imshow(visualization)
        plt.title(r"Correct: " + classes[true[i]] + '\n' + 'Pred: ' + classes[pred[i]])
        plt.xticks([])
        plt.yticks([])