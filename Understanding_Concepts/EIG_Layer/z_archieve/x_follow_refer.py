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

# def: Batch Normalization
class standardization_layer():

    def __init__(self): pass

    def feedforward(self,input,EPS=10e-5):
        self.input = input
        self.mean  = np.sum(input,axis=0) / input.shape[0]
        self.std   = np.sum( (self.input - self.mean) ** 2,0)  / input.shape[0]
        self.x_hat = (input - self.mean) / np.sqrt(self.std + EPS)
        return self.x_hat

    def backprop(self,grad,EPS=10e-5):
        dem = 1./(grad.shape[0] * np.sqrt(self.std + EPS ) )
        d_x = grad.shape[0] * grad - np.sum(grad,axis = 0) - self.x_hat*np.sum(grad*self.x_hat, axis=0)
        return d_x * dem

# def: zca whitening layer
class zca_whiten_layer():

    def __init__(self): pass

    def feedforward(self,input,EPS=10e-5):
        self.input = input
        self.sigma = input.T.dot(input) / input.shape[0]
        self.eigenval,self.eigvector = np.linalg.eigh(self.sigma)
        self.U = self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS))).dot(self.eigvector.T)
        self.whiten = input.dot(self.U)
        return self.whiten

    def backprop(self,grad,EPS=10e-5):
        d_U = self.input.T.dot(grad)
        d_eig_value = self.eigvector.T.dot(d_U).dot(self.eigvector) * (-0.5) * np.diag(1. / (self.eigenval+EPS) ** 1.5)
        d_eig_vector = d_U.dot( (np.diag(1. / np.sqrt(self.eigenval+EPS)).dot(self.eigvector.T)).T  ) + (self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)))).dot(d_U)
        E = np.ones((grad.shape[1],1)).dot(np.expand_dims(self.eigenval.T,0)) - np.expand_dims(self.eigenval  ,1).dot(np.ones((1,grad.shape[1])))
        K_matrix = 1./(E + np.eye(grad.shape[1])) - np.eye(grad.shape[1])
        np.fill_diagonal(d_eig_value,0.0)
        d_sigma = self.eigvector.dot(
                    K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + d_eig_value
                    ).dot(self.eigvector.T)
        d_x = grad.dot(self.U.T) + (2./grad.shape[0]) * self.input.dot(d_sigma) * 2
        return d_x

# data
# mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
mnist = input_data.read_data_sets('../../Dataset/fashionmnist/',one_hot=True)
train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# hyper
num_epoch = 30
batch_size = 20
learning_rate = 0.0003
print_size  = 1

beta1,beta2,adam_e = 0.9,0.999,10e-8

# class of layers
l0 = np_FNN(784,100,batch_size)
l0_special = zca_whiten_layer()
l1 = np_FNN(100,10 ,batch_size)

# train
for iter in range(num_epoch):

    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]

    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for current_data_index in range(0,len(train_data),batch_size):
        current_data = train_data[current_data_index:current_data_index+batch_size]
        current_label= train_label[current_data_index:current_data_index+batch_size]

        layer0 = l0.feedforward(current_data)
        layer0_special = l0_special.feedforward(layer0.T).T
        layer1 = l1.feedforward(layer0_special)

        cost = np.sum(layer1 - current_label)
        accuracy = np.mean(np.argmax(layer1,1) == np.argmax(current_label, 1))
        print('Current Iter: ', iter,' batch index: ', current_data_index, ' accuracy: ',accuracy, ' cost: ',cost,end='\r')
        train_cota = train_cota + cost; train_acca = train_acca + accuracy

        grad1 = l1.backprop(layer1 - current_label)
        grad0_special = l0_special.backprop(grad1.T).T
        grad0 = l0.backprop(grad1)

    for current_data_index in range(0,len(test_data),batch_size):
        current_data = test_data[current_data_index:current_data_index+batch_size]
        current_label= test_label[current_data_index:current_data_index+batch_size]

        layer0 = l0.feedforward(current_data)
        layer0_special = l0_special.feedforward(layer0.T).T
        layer1 = l1.feedforward(layer0_special)

        cost = np.sum(layer1 - current_label)
        accuracy = np.mean(np.argmax(layer1,1) == np.argmax(current_label, 1))
        print('Current Iter: ', iter,' batch index: ', current_data_index, ' accuracy: ',accuracy, ' cost: ',cost,end='\r')
        test_cota = test_cota + cost; test_acca = test_acca + accuracy

    if iter % print_size==0:
        print("\n----------")
        print('Train Current Acc: ', train_acca/(len(train_data)/batch_size),' Current cost: ', train_cota/(len(train_data)/batch_size),end='\n')
        print('Test  Current Acc: ', test_acca/(len(test_data)/batch_size),' Current cost: ', test_cota/(len(test_data)/batch_size),end='\n')
        print("----------")

    train_acc.append(train_acca/(len(train_data)/batch_size))
    train_cot.append(train_cota/(len(train_data)/batch_size))
    train_cota,train_acca = 0,0

    test_acc.append(test_acca/(len(test_data)/batch_size))
    test_cot.append(test_cota/(len(test_data)/batch_size))
    test_cota,test_acca = 0,0

    # shuffle the data
    # train_data,train_label = shuffle(train_data,train_label)



# -- end code --
