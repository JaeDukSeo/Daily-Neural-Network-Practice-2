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

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(789)

# import data
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
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


# create layer
def np_sigmoid(x): return 1.0 / (1.0+np.exp(-x))
def d_np_sigmoid(x): return np_sigmoid(x) * (1.0 - np_sigmoid(x))

# soft max function
def stable_softmax(X):
    exps = np.exp(X - np.max(X,axis=1)[:,np.newaxis])
    return exps / np.sum(exps)

# fully connected layer
class np_FNN():

    def __init__(self,inc,outc):
        self.w = np.random.randn(inc,outc)
        self.m,self.v = np.zeros_like(self.w),np.zeros_like(self.w)

    def feedforward(self,input):
        self.input  = input
        self.layer  = self.input.dot(self.w)
        self.layerA = np_sigmoid(self.layer)
        return self.layerA

    def backprop(self,grad):
        grad_1 = grad
        grad_2 = d_np_sigmoid(self.layer)
        grad_3 = self.input

        grad_middle = grad_1 * grad_2
        grad = grad_3.T.dot(grad_middle) / batch_size
        grad_pass = grad_middle.dot(self.w.T)

        self.m = self.m * beta1 + (1. - beta1) * grad
        self.v = self.v * beta2 + (1. - beta2) * grad ** 2
        m_hat,v_hat = self.m/(1.-beta1), self.v/(1.-beta2)
        adam_middle = learning_rate / (np.sqrt(v_hat) + adam_e) * m_hat
        self.w = self.w - adam_middle

        return grad_pass

# def: centering layer
class centering_layer():

    def __init__(self,batch_size):
        self.m = batch_size

    def feedforward(self,x):
        x_mean = np.sum(x,axis=0)
        return x - (1./self.m) * x_mean

    def backprop(self,grad):
        return grad * ( 1. - 1./self.m )

# def: whiten layer without centering
class zca_whiten_layer():

    def __init__(self,batch_size,feature_size):
        self.m = batch_size
        self.n = feature_size
        self.moving_sigma = 0
        self.moving_mean = 0

    def feedforward(self,input,EPS=1e-10):
        self.input = input
        self.sigma =  (1./self.m) * input.T.dot(input)
        self.eigenval,self.eigvector = np.linalg.eigh(self.sigma)
        self.U = self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)))
        self.whiten = input.dot(self.U)
        self.zca = self.whiten.dot(self.eigvector)
        return self.zca

    def backprop(self,grad,EPS=1e-10):
        d_eig_vector = self.whiten.T.dot(grad) + \
                       self.input.T.dot(grad.dot(self.eigvector.T)).dot(np.diag(1. / np.sqrt(self.eigenval+EPS)).T)

        d_eig_value  = self.input.T.dot(grad.dot(self.eigvector.T)) \
                     * (-1/2) * np.diag(1. / (self.eigenval+EPS) ** 1.5 )

        E = np.ones((self.n,1)).dot(np.expand_dims(self.eigenval.T,0)) - \
                   np.expand_dims(self.eigenval,1).dot(np.ones((1,self.n)))
        K_matrix = 1./(E + np.eye(self.n)) - np.eye(self.n)
        d_sigma = self.eigvector.dot(
                    K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + \
                    d_eig_value
                    ).dot(self.eigvector.T)

        d_simg_sym = (0.5) * (d_sigma.T + d_sigma)
        d_x = grad.dot(self.eigvector.T).dot(self.U.T) + \
              (2./self.m) * self.input.dot(d_simg_sym.T)
        return d_x

# def: Decorrelated Batch Normalization
class Decorrelated_Batch_Norm():

    def __init__(self,batch_size,feature_size):
        self.m = batch_size
        self.n = feature_size

    def feedforward(self,input,EPS=1e-10):
        self.input = input
        self.mean = (1./self.m) * np.sum(input,axis=0)
        self.sigma = (1./self.m) * (input - self.mean).T.dot(input - self.mean)
        self.eigenval,self.eigvector = np.linalg.eigh(self.sigma)
        self.U = self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)))
        self.whiten = (input-self.mean).dot(self.U)
        self.zca = self.whiten.dot(self.eigvector)
        return self.zca

    def backprop(self,grad,EPS=1e-10):

        d_eig_vector = self.whiten.T.dot(grad) + \
              (self.input-self.mean).T.dot(grad.dot(self.eigvector.T)).dot(np.diag(1. / np.sqrt(self.eigenval+EPS)).T)

        d_eig_value  = (self.input-self.mean).T.dot(grad.dot(self.eigvector.T)) \
                     * (-1/2) * np.diag(1. / (self.eigenval+EPS) ** 1.5 )

        E = np.ones((self.n,1)).dot(np.expand_dims(self.eigenval.T,0)) - \
                   np.expand_dims(self.eigenval,1).dot(np.ones((1,self.n)))
        K_matrix = 1./(E + np.eye(self.n)) - np.eye(self.n)
        d_sigma = self.eigvector.dot(
                    K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + \
                    d_eig_value
                    ).dot(self.eigvector.T)

        d_simg_sym = (0.5) * (d_sigma.T + d_sigma)
        d_mean = np.sum(grad.dot(self.eigvector.T).dot(self.U.T) * (-1),0) + \
                 (-2./self.m) *  np.sum((self.input-self.mean).dot(d_simg_sym.T),0)

        d_x = grad.dot(self.eigvector.T).dot(self.U.T) + \
              (2./self.m) * (self.input-self.mean).dot(d_simg_sym.T) + \
              (1./self.m) * d_mean
        return d_x

# class Batch Normalization
class Batch_Normalization_layer():

    def __init__(self,batch_size,feature_dim):
        self.m = batch_size
        self.moving_mean = np.zeros(feature_dim)
        self.moving_std  = np.zeros(feature_dim)

    def feedforward(self,input,EPS=1e-10):
        self.input = input
        self.mean  = (1./self.m) * np.sum(input,axis = 0 )
        self.std   = (1./self.m) * np.sum((self.input-self.mean) ** 2,axis = 0 )
        self.x_hat = (input - self.mean) / np.sqrt(self.std + EPS)
        return self.x_hat

    def backprop(self,grad,EPS=1e-10):
        dem = 1./(self.m * np.sqrt(self.std + EPS ) )
        d_x = self.m * grad - np.sum(grad,axis = 0) - self.x_hat*np.sum(grad*self.x_hat, axis=0)
        return d_x * dem

# hyper
num_epoch = 100
batch_size = 50

learning_rate = 0.00001
beta1,beta2,adam_e = 0.9,0.999,1e-8

l0 = np_FNN(784,400)
l1 = Decorrelated_Batch_Norm(batch_size,400)
l2 = np_FNN(400,256)
l3 = Decorrelated_Batch_Norm(batch_size,256)
l4 = np_FNN(256,100)
l5 = Decorrelated_Batch_Norm(batch_size,100)
l6 = np_FNN(100,10)

testing = train_data[:batch_size]
testing_label = train_label[:batch_size]

layer0 = l0.feedforward(testing)
layer1 = l1.feedforward(layer0)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)
layer4 = l4.feedforward(layer3)
layer5 = l5.feedforward(layer4)
layer6 = l6.feedforward(layer5)

final_soft = stable_softmax(layer6)
cost = np.mean(
  testing_label * np.log(final_soft + 1e-10) + \
  (1.0-testing_label) * np.log(1.0-final_soft + 1e-10)
  )
correct_prediction = np.equal(np.argmax(final_soft, 1), np.argmax(testing_label, 1))
accuracy = np.mean(correct_prediction)


print(layer6.shape)
print(final_soft.shape)
print(cost.shape)
print(correct_prediction.shape)
print(accuracy.shape)



















# -- end code --
