import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread,imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from imgaug import augmenters as iaa
import imgaug as ia
from scipy.ndimage import zoom
import seaborn as sns

np.random.seed(0)
np.set_printoptions(precision = 3,suppress =True)
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

# import data
# mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
mnist = input_data.read_data_sets('../../Dataset/fashionmnist/',one_hot=True)
train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Show some details and vis some of them
print(train_data.shape)
print(train_data.min(),train_data.max())
print(train_label.shape)
print(train_label.min(),train_label.max())
print(test_data.shape)
print(test_data.min(),test_data.max())
print(test_label.shape)
print(test_label.min(),test_label.max())
print('-----------------------')

# import layers
from x_layers import np_FNN,centering_layer,standardization_layer,zca_whiten_layer

# hyper
num_epoch = 100
batch_size = 100
print_size = 1

learning_rate = 0.0005
beta1,beta2,adam_e = 0.9,0.9,1e-8

# class of layers
one,two = 20,15
l1 = np_FNN(784,one*one)
l2 = np_FNN(one*one,two*two)
l3 = np_FNN(two*two,10)

# Def: shift the data into zero mean
centering_layer  = centering_layer()
# Def: shift the data into zero mean with unit variance
stand_layer = standardization_layer()
# Def: zca whiten the data
zca_layer  = zca_whiten_layer()
print(dir(centering_layer))
print(dir(stand_layer))
print(dir(zca_layer))

# train
for iter in range(num_epoch):

    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    # train_data,train_label = shuffle(train_data,train_label)

    for current_batch_index in range(0,len(train_data),batch_size):

        current_train_data = train_data[current_batch_index:current_batch_index + batch_size]
        current_train_data_label = train_label[current_batch_index:current_batch_index + batch_size]

        layer1 = l1.feedforward(current_train_data)
        layer1_center  = centering_layer.feedforward(layer1.T).T
        print(layer1_center[0].mean())
        print(layer1_center[0].std())
        print(layer1_center[0].max())
        print(layer1_center[0].min())
        print('========================')

        layer1_std  = stand_layer.feedforward(layer1.T).T
        print(layer1_std[0].mean())
        print(layer1_std[0].std())
        print(layer1_std[0].max())
        print(layer1_std[0].min())
        print('========================')

        layer1_zca  = zca_layer.feedforward(layer1.T).T
        print(layer1_zca[0].mean())
        print(layer1_zca[0].std())
        print(layer1_zca[0].max())
        print(layer1_zca[0].min())
        print('========================')

        layer1_std_zca  = zca_layer.feedforward(layer1_std.T).T
        print(layer1_std_zca[0].mean())
        print(layer1_std_zca[0].std())
        print(layer1_std_zca[0].max())
        print(layer1_std_zca[0].min())
        print('========================')


        sys.exit()



# -- end code --
