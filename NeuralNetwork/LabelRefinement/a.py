import tensorflow as tf
import numpy as np,sys,os
from numpy import float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize

np.random.seed(678)
tf.set_random_seed(678)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),tf.float32)
def tf_LRelu(x): return tf.nn.leaky_relu(x)
def d_tf_LRelu(x): return tf.cast(tf.greater(x,0),tf.float32) + tf.cast(tf.less_equal(x,0),tf.float32) *x* 0.2
def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))
def tf_log(x): return tf.nn.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1-tf_log(x))

# convolution layer
class CNNLayer():
    
    def __init__(self,ker,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.02))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
    def getw(self): return [self.w]
    def feedforward(self,input,stride=2,batch_norm=True,padding_val='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding=padding_val)
        if batch_norm: self.layer = tf.nn.batch_normalization(self.layer,mean=0.0,variance=1.0,variance_epsilon=1e-8,scale=True,offset=True)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient,stride=1):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,filter_sizes = self.w.shape,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        grad_pass  = tf.nn.conv2d_backprop_input(
            input_sizes=[batch_size] + list(self.input.shape[1:]),filter = self.w ,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        update_w = []

        update_w.append(
            tf.assign( self.m,self.m*beta_1 + (1-beta_1) * grad   )
        )
        update_w.append(
            tf.assign( self.v,self.v*beta_2 + (1-beta_2) * grad ** 2   )
        )

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,update_w



# data

# hyper

# define class
l1_e = CNNLayer()
l2_e = CNNLayer()
l3_e = CNNLayer()
l4_e = CNNLayer()
l5_e = CNNLayer()



# graph

# session


# -- end code --