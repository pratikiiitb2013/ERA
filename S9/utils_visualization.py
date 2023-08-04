import numpy as np
import matplotlib.pyplot as plt

def visualize_data_loader(images, labels, classes, grid_size):
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    for i in range(0,25):
        plt.subplot(grid_size, grid_size, i+1)
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])
        plt.imshow(np.transpose(((images[i]/2)+0.5).numpy(),(1,2,0)))
        plt.title(classes[labels[i]])

