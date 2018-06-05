import tensorflow as tf,numpy as np,pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(678)
tf.set_random_seed(678)

# data
mnist = input_data.read_data_sets("../../Dataset/MNIST/", one_hot=True)
train_image,train_label = mnist.train.images,mnist.train.labels
test_image ,test_label  = mnist.test.images,mnist.test.labels

# class

# train 




# -- end code --