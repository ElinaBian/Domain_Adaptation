import scipy
import matplotlib.pyplot as plt
#%matplotlib inline
from skimage import color

def plot_images(img, nrows, ncols):
    #Plot nrows x ncols images
    img_2 = img.copy()
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img_2[i].shape == (32, 32, 3):
            img_2[i] = img_2[i]
            ax.imshow(img_2[i])
        else:
            ax.imshow(img_2[i,:,:,0])
        ax.set_xticks([])
        ax.set_yticks([])
        
    