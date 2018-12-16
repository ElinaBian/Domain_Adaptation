import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage import color, transform
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import LabelBinarizer


def normalize(x, max_value):
    """ If x takes its values between 0 and max_value, normalize it between -1 and 1"""
    return (x / float(max_value)) * 2 - 1

def transform_mnist(X):
    X = X.reshape(len(X), 28, 28)
    X = np.array([transform.resize(im, [32,32,1]) for im in X])
    X = normalize(X, 1)
    #X = X.reshape(len(X), 1024)
    return X

def load_mnist_data(num_pixel):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist_train_images = mnist.train.images
    mnist_test_images = mnist.test.images
    mnist_train_labels = mnist.train.labels
    mnist_test_labels = mnist.test.labels
    if num_pixel == 32:
        mnist_train_images = transform_mnist(mnist.train.images)
        mnist_test_images = transform_mnist(mnist.test.images)
        lb_mnist = LabelBinarizer()
        mnist_train_labels = lb_mnist.fit_transform(mnist_train_labels)
        mnist_test_labels = lb_mnist.fit_transform(mnist_test_labels)
        return mnist_train_images, mnist_train_labels, mnist_test_images, mnist_test_labels