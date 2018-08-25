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

# def: relu activations
def np_relu(x): return x * (x > 0)
def d_np_relu(x): return 1. * (x > 0)

# def: fully connected layer
r = np.random.RandomState(1234)
class np_FNN():

    def __init__(self,inc,outc,batch_size,act=np_relu,d_act = d_np_relu):
        self.w = r.normal(0,0.05,size=(inc, outc))
        self.b = np.zeros(outc)
        self.m,self.v = np.zeros_like(self.w),np.zeros_like(self.w)
        self.mb,self.vb = np.zeros_like(self.b),np.zeros_like(self.b)
        self.act = act; self.d_act = d_act
        self.batch_size = batch_size

    def getw(self): return self.w

    def feedforward(self,input):
        self.input  = input
        self.layer  = self.input.dot(self.w) + self.b
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,grad):
        grad_1 = grad
        grad_2 = self.d_act(self.layer)
        grad_3 = self.input

        grad_middle = grad_1 * grad_2
        grad_b = grad_middle.mean(0)
        grad = grad_3.T.dot(grad_middle) / self.batch_size
        # self.w = self.w - learning_rate * grad

        grad_pass = grad_middle.dot(self.w.T)

        self.m = self.m * beta1 + (1. - beta1) * grad
        self.v = self.v * beta2 + (1. - beta2) * grad ** 2
        m_hat,v_hat = self.m/(1.-beta1), self.v/(1.-beta2)
        adam_middle =  m_hat * learning_rate / (np.sqrt(v_hat) + adam_e)
        self.w = self.w - adam_middle

        self.mb = self.mb * beta1 + (1. - beta1) * grad_b
        self.vb = self.vb * beta2 + (1. - beta2) * grad_b ** 2
        m_hatb,v_hatb = self.mb/(1.-beta1), self.vb/(1.-beta2)
        adam_middleb =  m_hatb *learning_rate /(np.sqrt(v_hatb) + adam_e)
        self.b = self.b - adam_middleb
        return grad_pass

# data
# mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
mnist = input_data.read_data_sets('../../Dataset/fashionmnist/',one_hot=True)
train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# hyper
num_epoch = 30
batch_size = 20
learning_rate = 0.0002

beta1,beta2,adam_e = 0.9,0.999,10e-8

l0 = np_FNN(784,100,batch_size)
l1 = np_FNN(100,10 ,batch_size)

for iter in range(num_epoch):

    for current_data_index in range(0,len(trX),batch_size):
        current_data = train_data[current_data_index:current_data_index+batch_size]
        current_label= train_label[current_data_index:current_data_index+batch_size]

        layer0 = l0.feedforward(current_data)
        layer1 = l1.feedforward(layer0)

        cost = np.sum(layer1 - current_label)
        accuracy = np.mean(np.argmax(layer1,1) == np.argmax(current_label, axis=1))
        print('Current Iter: ', iter,' batch index: ', current_data_index, ' accuracy: ',accuracy, ' cost: ',cost,end='\r')

        grad1 = l1.backprop(layer1 - current_label)
        grad0 = l0.backprop(grad1)

    test_predict = np.argmax(l1.feedforward(l0.feedforward(test_data)),1)
    print('\n-------------')
    print(iter, np.mean(test_predict == np.argmax(test_label, axis=1)))
    print('-------------')


# -- end code --
