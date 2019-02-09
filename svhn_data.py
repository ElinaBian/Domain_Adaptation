from scipy.io import loadmat
import numpy as np

def indices_to_one_hot(data, nb_classes=10):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def load_svhn_data(hot=True):
    """ Helper function for loading a MAT-File"""
    train_data = loadmat('./data/svhn_train_32x32.mat')
    test_data = loadmat('./data/svhn_test_32x32.mat')
    
    #transform train data
    train_images = train_data['X']
    train_images = train_images.transpose((3,0,1,2))
    train_labels = train_data['y']
    train_labels = train_labels[:,0]
    train_labels[train_labels == 10] = 0
    train_labels_hot = indices_to_one_hot(train_labels.astype(int), 10)
    
    #transform test dat
    test_images = test_data['X']
    test_images = test_images.transpose((3,0,1,2))
    test_labels = test_data['y']
    test_labels = test_labels[:,0]
    test_labels[test_labels == 10] = 0
    test_labels_hot = indices_to_one_hot(test_labels.astype(int), 10)
    
    if hot ==  True:
        return train_images, train_labels_hot, test_images, test_labels_hot
    else:
        return train_images, train_labels, test_images, test_labels
def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)