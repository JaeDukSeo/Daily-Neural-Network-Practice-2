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
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
# mnist = input_data.read_data_sets('../../Dataset/fashionmnist/',one_hot=True)
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
from x_layers import np_FNN,centering_layer,standardization_layer,zca_whiten_layer,stable_softmax

# hyper
num_epoch = 100
batch_size = 20
print_size = 1

# class of layers
one,two = 14,7
l1 = np_FNN(784,one*one,batch_size=batch_size)
l2 = np_FNN(one*one,two*two,batch_size=batch_size)
l3 = np_FNN(two*two,10,batch_size=batch_size)

# Def: shift the data into zero mean
center_layer  = centering_layer()
# Def: shift the data into zero mean with unit variance
stand_layer_1 = standardization_layer()
stand_layer_2 = standardization_layer()
# Def: zca whiten the data
zca_layer_1  = standardization_layer()
zca_layer_2  = standardization_layer()

# train
for iter in range(num_epoch):

    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    # train_data,train_label = shuffle(train_data,train_label)

    for current_batch_index in range(0,len(train_data),batch_size):

        current_train_data = train_data[current_batch_index:current_batch_index + batch_size]
        current_train_data_label = train_label[current_batch_index:current_batch_index + batch_size]

        layer1 = l1.feedforward(current_train_data)
        # layer1_std = stand_layer_1.feedforward(layer1.T).T
        # layer1_zca = zca_layer_1.feedforward(layer1_std.T).T

        layer2 = l2.feedforward(layer1)
        # layer2_std = stand_layer_2.feedforward(layer2.T).T
        # layer2_zca = zca_layer_2.feedforward(layer2_std.T).T

        layer3 = l3.feedforward(layer2)

        # cost
        final_soft = stable_softmax(layer3)
        cost = - np.mean(current_train_data_label * np.log(final_soft + 1e-20) + (1.0-current_train_data_label) * np.log(1.0-final_soft + 1e-20))
        correct_prediction = np.equal(np.argmax(final_soft, 1), np.argmax(current_train_data_label, 1))
        accuracy = np.mean(correct_prediction)
        print('Current Iter: ', iter,' batch index: ', current_batch_index, ' accuracy: ',accuracy, ' cost: ',cost,end='\r')
        train_cota = train_cota + cost
        train_acca = train_acca + accuracy

        # print('\n--------------------')
        # print( (final_soft-current_train_data_label).sum() )
        # print( (final_soft-current_train_data_label).mean() )
        # print( final_soft.sum(1) )
        # input()
        # print('--------------------\n')

        # back prop
        grad3 = l3.backprop(final_soft-current_train_data_label)

        # grad_zca_2 = zca_layer_2.backprop(grad3.T).T
        # grad_std_2 = stand_layer_2.backprop(grad_zca_2.T).T
        grad2 = l2.backprop(grad3)

        # grad_zca_1 = zca_layer_1.backprop(grad2.T).T
        # grad_std_1 = stand_layer_1.backprop(grad_zca_1.T).T
        grad1 = l1.backprop(grad2)

    if iter % print_size==0:
        print("\n----------")
        print('Train Current Acc: ', train_acca/(len(train_data)/batch_size),' Current cost: ', train_cota/(len(train_data)/batch_size),end='\n')
        print("----------")

    train_acc.append(train_acca/(len(train_data)/batch_size))
    train_cot.append(train_cota/(len(train_data)/batch_size))
    train_cota,train_acca = 0,0




# -- end code --
