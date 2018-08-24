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

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(789)


def np_sig(x): return 1.0/(1.0+np.exp(-x))
def d_np_sig(x): return np_sig(x) * (1.0 - np_sig(x))

class FNN_numpy():

    def __init__(self,inc,outc):
        self.w = np.random.randn(inc,outc)
        self.m,self.v = np.zeros_like(self.w),np.zeros_like(self.w)

    def feedforward(self,input):
        self.input = input
        self.layer = self.input.dot(self.w)
        self.layerA = np_sig(self.layer)
        return self.layerA

    def backprop(self,grad):
        grad_part_1 = grad
        grad_part_2 = d_np_sig(self.layer)
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2
        grad = grad_part_3.T.dot(grad_middle)
        grad_pass = grad_middle.dot(self.w.T)

        self.m = self.m * beta1 + (1.0-beta1) * grad
        self.v = self.v * beta2 + (1.0-beta2) * grad ** 2
        m_hat,v_hat = self.m/(1.-beta1),self.v/(1.-beta2)
        adam_mid = learning_rate / (np.sqrt(v_hat) + adam_e) * m_hat
        self.w = self.w - adam_mid
        return grad_pass
class whitening_layer():

    def __init__(self):
        self.moving_sigma = 0
        self.moving_mean = 0

    def feedforward(self,input,is_training=True):
        self.input = input
        self.mean = np.mean(input,axis=0)
        self.sigma = (input - self.mean).T.dot(input - self.mean) / input.shape[1]
        self.eigenval,self.eigvector = np.linalg.eigh(self.sigma)
        self.U = self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval + white_e)))
        self.whiten = (input - self.mean).dot(self.U)
        self.zca = self.whiten.dot(self.eigvector)
        return self.zca

    def backprop(self,grad):
        dwhiten = grad.dot(self.eigvector.T)
        dU = (self.input-self.mean).T.dot(dwhiten)
        deigenval = self.eigvector.T.dot(dU).dot(np.diag(-0.5 * 1/(self.eigenval ** 1.5)))
        deigvector = self.whiten.T.dot(grad) + dU.dot(np.diag(1. / np.sqrt(self.eigenval + white_e) ).T)

        shape_eig = self.eigenval.shape[0]
        Iden = np.eye(shape_eig)
        E = np.ones((shape_eig,1)).dot(np.expand_dims(self.eigenval.T,0))-np.expand_dims(self.eigenval,1).dot(np.ones((1,shape_eig)))
        F = 1.0/(E+Iden) - Iden
        deigenval_zero = np.eye(deigenval.shape[0]) * deigenval
        dsigma = self.eigvector.dot( F.T * (self.eigvector.dot(deigvector))+deigenval_zero).dot(self.eigvector.T)

        symm_sigma = 0.5 * (dsigma.T + dsigma)
        dmean = -np.sum(dwhiten,0).dot(self.U.T) + (-2/shape_eig) * np.mean(self.input - self.mean,0).dot(symm_sigma)

        dInput = dwhiten.dot(self.U.T) + (2/shape_eig) * (self.input - self.mean).dot(symm_sigma) + 1/shape_eig * dmean
        return dInput

# # data
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
x_data, train_label, y_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
x_data_added,x_data_added_label = mnist.validation.images,mnist.validation.labels

x_data = np.vstack((x_data,x_data_added))
train_batch = x_data
train_label = np.vstack((train_label,x_data_added_label))
test_batch = y_data

# print out the data shape and the max and min value
print(train_batch.shape)
print(train_batch.max())
print(train_batch.min())
print(train_label.shape)
print(train_label.max())
print(train_label.min())
print(test_batch.shape)
print(test_batch.max())
print(test_batch.min())
print(test_label.shape)
print(test_label.max())
print(test_label.min())

# hyper
num_epoch = 100
batch_size = 100

learning_rate = 0.0001
beta1,beta2,adam_e = 0.9,0.999,1e-8
white_e = 1e-10

# class
l0 = FNN_numpy(784,256)
l1 = whitening_layer()
l2 = FNN_numpy(256,10)

# train
for iter in range(num_epoch):

    for current_batch_index in range(0,len(train_batch),batch_size):

        current_data = train_batch[current_batch_index:current_batch_index+batch_size]
        current_data_label = train_label[current_batch_index:current_batch_index+batch_size]

        layer0 = l0.feedforward(current_data)
        layer1 = l1.feedforward(layer0)
        layer2 = l2.feedforward(layer1)
        sys.exit()



# -- end code --
