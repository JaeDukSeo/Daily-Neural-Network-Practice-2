import tensorflow as tf
import numpy as np
import sys, os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
from skimage.transform import resize

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(x): return tf.cast(tf.greater(x,0),tf.float32)
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32) + tf_elu(tf.cast(tf.less_equal(x,0),tf.float32)*x)

np.random.seed(676)
tf.set_random_seed(6787)

# data

# class
class CNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.005))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,maxpool=False):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerAa = tf_leaky_relu(self.layer) 
        self.layerAb = tf_leaky_relu(-1.0 * self.layer)
        self.out = tf.nn.avg_pool(tf.concat([self.layerAa ,self.layerAb],3), [ 1, 2, 2, 1 ], [1, 2, 2, 1 ], 'VALID')
        return self.out

    def backprop(self,gradient):
        half_shape = gradient.shape[3].value//2
        gradient = tf_repeat(gradient,[1,2,2,1])
        
        grad_part_1 = gradient 
        grad_part_2a = d_leaky_tf_relu(self.layer) 
        grad_part_2b = d_leaky_tf_relu(-1.0 * self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1[:,:,:,:half_shape] * grad_part_2b + grad_part_1[:,:,:,half_shape:] * grad_part_2a

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
            filter= self.w,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))
        
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)

        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,grad_update     

class DeCNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.005))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,de_stride = 2):
        self.input  = input
        self.layer = tf.nn.conv2d_transpose(input,self.w,output_shape=[1,shape,shape,1],strides=[1,de_stride,de_stride,1],padding='SAME')
        return self.layer

    def backprop(self,gradient):
        half_shape = gradient.shape[3].value//2
        gradient = tf_repeat(gradient,[1,2,2,1])
        
        grad_part_1 = gradient 
        grad_part_2a = d_leaky_tf_relu(self.layer) 
        grad_part_2b = d_leaky_tf_relu(-1.0 * self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1[:,:,:,:half_shape] * grad_part_2b + grad_part_1[:,:,:,half_shape:] * grad_part_2a

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
            filter= self.w,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))
        
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)

        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,grad_update     


# hyper


# define class

# graph

# session




# -- end code --