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
r = np.random.RandomState(1234)
np.set_printoptions(precision = 3,suppress =True)
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
from scipy import linalg as LA
# def: activations
def np_relu(x): return x * (x > 0)
def d_np_relu(x): return 1. * (x > 0)
def np_tanh(x): return  np.tanh(x)
def d_np_tanh(x): return 1. - np_tanh(x) ** 2
def np_sigmoid(x): return  1/(1+np.exp(-x))
def d_np_sigmoid(x): return np_sigmoid(x) * (1.-np_sigmoid(x))

# def: fully connected layer
class np_FNN():

    def __init__(self,inc,outc,act=np_relu,d_act = d_np_relu):
        self.w = r.normal(0.0,0.008,size=(inc, outc)).astype(np.float64)
        # self.b = r.normal(0,0.005,size=(outc))
        self.b = np.zeros(outc).astype(np.float64)
        self.m,self.v = np.zeros_like(self.w),np.zeros_like(self.w)
        self.mb,self.vb = np.zeros_like(self.b),np.zeros_like(self.b)
        self.act = act; self.d_act = d_act

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
        grad = grad_3.T.dot(grad_middle) / grad.shape[0]
        grad_pass = grad_middle.dot(self.w.T)

        self.m = self.m * beta1 + (1. - beta1) * grad
        self.v = self.v * beta2 + (1. - beta2) * grad ** 2
        m_hat,v_hat = self.m/(1.-beta1), self.v/(1.-beta2)
        adam_middle =  m_hat * learning_rate / (np.sqrt(v_hat) + adam_e)
        self.w = self.w - adam_middle

        self.mb = self.mb * beta1 + (1. - beta1) * grad_b
        self.vb = self.vb * beta2 + (1. - beta2) * grad_b ** 2
        m_hatb,v_hatb = self.mb/(1.-beta1), self.vb/(1.-beta2)
        adam_middleb =  m_hatb * learning_rate /(np.sqrt(v_hatb) + adam_e)
        self.b = self.b - adam_middleb
        return grad_pass

# def: centering layer
class centering_layer():

    def __init__(self,batch_size):
        self.moving_mean = np.zeros(batch_size)

    def feedforward(self,input,is_training=True):
        if is_training:
            x_mean = np.sum(input,axis=0) / input.shape[0]
            self.moving_mean = self.moving_mean * 0.9 + (1-0.9) * x_mean
            return input - x_mean
        else:
            return input - self.moving_mean

    def backprop(self,grad):
        return grad * ( 1. - 1./grad.shape[0])

# def: Batch Normalization
class standardization_layer():

    def __init__(self): pass

    def feedforward(self,input,EPS=1e-15,is_training = True):
        self.input = input
        self.mean  = np.sum(input,axis=0) / input.shape[0]
        self.std   = np.sum( (self.input - self.mean) ** 2,0)  / input.shape[0]
        self.moving_mean = self.moving_mean * 0.9 + (1-0.9) * self.mean
        self.moving_std = self.moving_std * 0.9 + (1-0.9) * self.std
        self.x_hat = (input - self.mean) / np.sqrt(self.std + EPS)
        return self.x_hat

    def backprop(self,grad,EPS=1e-15):
        dem = 1./(grad.shape[0] * np.sqrt(self.std + EPS ) )
        d_x = grad.shape[0] * grad - np.sum(grad,axis = 0) - self.x_hat*np.sum(grad*self.x_hat, axis=0)
        return d_x * dem

# def: zca whitening layer
class zca_whiten_layer():

    def __init__(self): pass

    def feedforward(self,input,EPS=1e-11):
        self.input = input
        self.sigma = input.T.dot(input) / input.shape[0]

        # numpy eigh is not the best choice: https://stackoverflow.com/questions/6684238/whats-the-fastest-way-to-find-eigenvalues-vectors-in-python
        # self.eigenval,self.eigvector = np.linalg.eigh(self.sigma)
        # scipy eigh is more stable
        self.eigenval,self.eigvector = LA.eigh(self.sigma)
        self.U = self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS) )).dot(self.eigvector.T)
        self.whiten = input.dot(self.U)
        return self.whiten

    def backprop(self,grad,EPS=1e-11):
        d_U = self.input.T.dot(grad)
        d_eig_value = self.eigvector.T.dot(d_U).dot(self.eigvector) * (-0.5) * np.diag(1. / (self.eigenval+EPS) ** 1.5)
        d_eig_vector = d_U.dot( (np.diag(1. / np.sqrt(self.eigenval+EPS)).dot(self.eigvector.T)).T  ) + (self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)))).dot(d_U)
        E = np.ones((grad.shape[1],1)).dot(np.expand_dims(self.eigenval.T,0)) - np.expand_dims(self.eigenval,1).dot(np.ones((1,grad.shape[1])))
        K_matrix = 1./(E + np.eye(grad.shape[1])) - np.eye(grad.shape[1])
        np.fill_diagonal(d_eig_value,0.0)
        d_sigma = self.eigvector.dot(K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + d_eig_value).dot(self.eigvector.T)
        d_sigma_smooth = (0.5) * (d_sigma.T + d_sigma)
        d_x = grad.dot(self.U.T) + (2./grad.shape[0]) * self.input.dot(d_sigma_smooth)
        return d_x

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

# hyper
num_epoch = 50
batch_size = 250
learning_rate = 0.0008
print_size  = 1
beta1,beta2,adam_e = 0.9,0.999,1e-40

# class of layers
l0 = np_FNN(28*28,30*30,act=np_relu,d_act=d_np_relu)
l0_zca = zca_whiten_layer()
l1 = np_FNN(30*30,32*32 ,act=np_relu,d_act=d_np_relu)
l2 = np_FNN(32*32,10    ,act=np_relu,d_act=d_np_relu)

# train
train_cota,train_acca = 0,0; train_cot,train_acc = [],[]
test_cota,test_acca = 0,0; test_cot,test_acc = [],[]
for iter in range(num_epoch):

    # train_data,train_label = shuffle(train_data,train_label)
    # train data set run network
    for current_data_index in range(0,len(train_data),batch_size):
        current_data = train_data[current_data_index:current_data_index+batch_size].astype(np.float64)
        current_label= train_label[current_data_index:current_data_index+batch_size].astype(np.float64)

        layer0 = l0.feedforward(current_data)
        layer0_special_1 = l0_zca.feedforward(layer0.T).T
        layer1 = l1.feedforward(layer0_special_1)
        layer2 = l2.feedforward(layer1)

        cost = np.mean( layer2 - current_label )
        accuracy = np.mean(np.argmax(layer2,1) == np.argmax(current_label, 1))
        print('Current Iter: ', iter,' batch index: ', current_data_index, ' accuracy: ',accuracy, ' cost: ',cost,end='\r')
        train_cota = train_cota + cost; train_acca = train_acca + accuracy

        grad2 = l2.backprop(layer2 - current_label)
        grad1 = l1.backprop(grad2)
        grad0_special_1 = l0_zca.backprop(grad1.T).T
        grad0 = l0.backprop(grad0_special_1)

    # test data set run network
    for current_data_index in range(0,len(test_data),batch_size):
        current_data = test_data[current_data_index:current_data_index+batch_size].astype(np.float64)
        current_label= test_label[current_data_index:current_data_index+batch_size].astype(np.float64)

        layer0 = l0.feedforward(current_data)
        layer0_special_1 = l0_zca.feedforward(layer0.T).T
        layer1 = l1.feedforward(layer0_special_1)
        layer2 = l2.feedforward(layer1)

        cost = np.mean( layer2 - current_label )
        accuracy = np.mean(np.argmax(layer2,1) == np.argmax(current_label, 1))
        print('Current Iter: ', iter,' batch index: ', current_data_index, ' accuracy: ',accuracy, ' cost: ',cost,end='\r')
        test_cota = test_cota + cost; test_acca = test_acca + accuracy

    # print the results
    if iter % print_size==0:
        print("\n----------")
        print('Train Current Acc: ', train_acca/(len(train_data)/batch_size),' Current cost: ', train_cota/(len(train_data)/batch_size),end='\n')
        print('Test  Current Acc: ', test_acca/(len(test_data)/batch_size),' Current cost: ', test_cota/(len(test_data)/batch_size),end='\n')
        print("----------")

    # append the results
    train_acc.append(train_acca/(len(train_data)/batch_size))
    train_cot.append(train_cota/(len(train_data)/batch_size))
    train_cota,train_acca = 0,0
    test_acc.append(test_acca/(len(test_data)/batch_size))
    test_cot.append(test_cota/(len(test_data)/batch_size))
    test_cota,test_acca = 0,0


# -- end code --
