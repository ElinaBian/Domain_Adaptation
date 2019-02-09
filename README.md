# ElinaBian

Maximum Classifier Discrepency for Unsupervised Domain Adaptation
===========

Data Download
-----------

We used three datasets for our experiments. 

The MNIST dataset could be accessed through tensorflow tutorial, so it doesn't need to be download.

The USPS and SVHN data could be download [here](https://drive.google.com/file/d/1x3jVXOU0iQ0Wbli7ZeqD7ijjxFL9z3Y-/view?usp=sharing).

You should download the data using ./data as the path.

Data Transforming
---------------

There are four python files in ./data_process. 

1) mnist_data.py: This file could help us load MNIST data into appropriate format, like 28*28 for USPS to MNIST experiment and 32*32 for SVHN to MNIST experiment.

2) svhn_data.py: This file could help us load SVHN data into appropriate format and transform images into grey scale.

3) usps_data.py: This file could help us load USPS data into appropirate format.

4) visualization.py: This file could help us visualize the data.

Training Process
--------------

For python files:

1) mcd_usps2mnist.py: This file contains the main training process of domain adaptation on USPS to MNIST using Adam optimizer.

2) mcd_usps2mnist_Mom.py: This file contains the main training process of domain adaptation on USPS to MNIST using SGD with Momentum as optimizer.

3) mcd_svhn2mnist.py: This file contains the main training process of domain adaptation on SVHN to MNIST using Adam optimizer.

4) mcd_svhn2mnist_Mom.py: This file contains the main training process of domain adaptation on SVHN to MNIST using Adam optimizer.

For jupyter notebooks:

1) usps_mnist_n=2.ipynb: This file contains our best results for num_stepC=2, with Adam optimizer, batch_size = 64, epochs = 200, learning_rate = 0.0002.

2) usps_mnist_n=3.ipynb: This file contains our best results for num_stepC=3, with SGD Momentum optimizer, batch_size = 32, epochs = 200, learning_rate = 0.0002.

3) usps_mnist_n=4.ipynb: This file contains our best results for num_stepC=4, with Adam optimizer, batch_size = 32, epochs = 200, learning_rate = 0.0001.

4) svhn_mnist_n=2.ipynb: This file contains our best results for num_stepC=2, with Adam optimizer, batch_size = 64, epochs = 20, learning_rate = 0.0002.

5) svhn_mnist_n=3.ipynb: This file contains our best results for num_stepC=3, with SGD Momentum optimizer, batch_size = 64, epochs = 40, learning_rate = 0.0002.

6) svhn_mnist_n=4.ipynb: This file contains our best results for num_stepC=4, with batch_size = 32, epochs = 20, learning_rate = 0.0002.

Saving Model
-----------

In ./usps_model, we saved best models for our USPS to MNIST experiment.

Model named MCD_Usps2Mnist_1545015072 is the best model for USPS to MNIST when num_stepC = 2.

Model named MCD_Usps2Mnist_1545016343 is the best model for USPS to MNIST when num_stepC = 3.

Model named MCD_Usps2Mnist_1545019791 is the best model for USPS to MNIST when num_stepC = 4.
