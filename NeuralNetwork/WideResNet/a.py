import tensorflow as tf
import numpy as np
import sys, os
import cv2
from sklearn.utils import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
from skimage.transform import resize

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(x): return tf.cast(tf.greater(x,0),tf.float32)
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32) + tf_elu(tf.cast(tf.less_equal(x,0),tf.float32)*x)




# -- end code --