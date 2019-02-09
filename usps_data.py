from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from skimage import color, transform

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def load_usps_train_data(onehot = True):
    path = './data/usps_train'
    num_data = 20000
    num_class = 10
    paths = []
    for i in np.arange(num_class):
        paths.append("{}/{}/*.png".format(path, i))
        
    X_train = np.empty((num_data,28,28))
    y_train = np.empty(num_data)
    
    sub_num = int(num_data/num_class)
    
    for i, path in enumerate(paths):
        filenames = glob.glob(path)
        temp = np.empty((len(filenames), 28,28))
        print(str(len(filenames)) + ' images in class{}'.format(i)  )
        for j, fname in enumerate(filenames):
            img = Image.open(fname)
            img = img.convert('1')
            img = img.resize((28,28))
            arr = np.array(img)
            X_train[(sub_num*i) + j] = arr/255
            y_train[(sub_num*i) + j] = i  
    
    if onehot == True:
        y_train = indices_to_one_hot(y_train.astype(int), num_class)
        
    # delete #5999 since theres's only 1999 images in class 2
    X_train = np.delete(X_train, 5999, axis=0)
    y_train = np.delete(y_train, 5999, axis=0)
        
    return X_train, y_train
    
    
def load_usps_test_data(onehot = True):
    path = './data/usps_test'
    num_data = 1500
    num_class = 10
    paths = []
    for i in np.arange(num_class):
        paths.append("{}/{}/*.png".format(path, i))
        
    X_test = np.empty((num_data,28,28))
    y_test = np.empty(num_data)
    
    sub_num = int(num_data/num_class)
    
    for i, path in enumerate(paths):
        filenames = glob.glob(path)
        temp = np.empty((len(filenames), 28,28))
        print(str(len(filenames)) + ' images in class{}'.format(i)  )
        for j, fname in enumerate(filenames):
            img = Image.open(fname)
            img = img.convert('1')
            img = img.resize((28,28),Image.ANTIALIAS)
            arr = np.array(img)
            X_test[(sub_num*i) + j] = arr/255
            y_test[(sub_num*i) + j] = i  
    
    if onehot == True:
        y_test= indices_to_one_hot(y_test.astype(int), num_class)
        
    return X_test, y_test
    
    
    
