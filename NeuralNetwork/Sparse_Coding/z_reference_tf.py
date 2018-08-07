# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:54:49 2014
@author: Paul Rothnie
email : paul.rothnie@googlemail.com
Replicates the sparse autoencoder exercises from the ufldl tutorial on 
http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
and
http://ufldl.stanford.edu/wiki/index.php/Exercise:Vectorization
Uses some of the data loading and visualization functions written by Siddharth 
Agrawal for the same exercise.  His github page is
https://github.com/siddharth950/Sparse-Autoencoder
"""

import numpy as np
import numpy.linalg as la
import scipy.io
import scipy.optimize
import matplotlib.pyplot
import time
import struct
import array
import matplotlib.pyplot as plt

import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Generate training data 
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data


class sparse_autoencoder(object):
    
    def __init__(self, visible_size, hidden_size, lambda_, rho, beta):
        
        self.visible_size = visible_size  
        self.hidden_size = hidden_size 
        self.lambda_ = lambda_ 
        self.rho = rho 
        self.beta = beta 
                
        # initialize weights and bias terms 
        w_max = np.sqrt(6.0 / (visible_size + hidden_size + 1.0))
        w_min = -w_max
        W1 = (w_max - w_min) * np.random.random_sample(size = (hidden_size, visible_size)) + w_min
        W2 = (w_max - w_min) * np.random.random_sample(size = (visible_size, hidden_size)) + w_min
        b1 = np.zeros(hidden_size)
        b2 = np.zeros(visible_size)
        
        # unroll the weights and bias terms into an initial "guess" for theta 
        # (solver expects a vector)
        self.idx_0 = 0
        self.idx_1 = hidden_size * visible_size # length of W1
        self.idx_2 = self.idx_1 +  hidden_size * visible_size # length of W2
        self.idx_3 = self.idx_2 + hidden_size # length of b1
        self.idx_4 = self.idx_3 + visible_size # length of b2
        self.initial_theta = np.concatenate((W1.flatten(), W2.flatten(), 
                                             b1.flatten(), b2.flatten()))
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def feedforward(self,input,theta):
        # Retrieve the weights and biases from theta.        
        W1, W2, b1, b2 = self.unpack_theta(self.)
        hidden_layer = self.sigmoid(np.dot(W1, input) + b1)
        output_layer = self.sigmoid(np.dot(W2, hidden_layer) + b2)
        return output_layer
    
    def unpack_theta(self, theta):
        W1 = theta[self.idx_0 : self.idx_1]
        W1 = np.reshape(W1, (self.hidden_size, self.visible_size))
        W2 = theta[self.idx_1 : self.idx_2]
        W2 = np.reshape(W2, (self.visible_size, self.hidden_size))
        b1 = theta[self.idx_2 : self.idx_3]
        b1 = np.reshape(b1, (self.hidden_size, 1))
        b2 = theta[self.idx_3 : self.idx_4]
        b2 = np.reshape(b2, (self.visible_size, 1))
        return W1, W2, b1, b2     

    def cost(self, theta, visible_input):
        # Retrieve the weights and biases from theta.        
        W1, W2, b1, b2 = self.unpack_theta(theta)
        
        # Forward pass to get the activation levels.        
        hidden_layer = self.sigmoid(np.dot(W1, visible_input) + b1)
        output_layer = self.sigmoid(np.dot(W2, hidden_layer) + b2)
        m = visible_input.shape[1] # number of training examples
        
        # Calculate the cost.         
        error = -(visible_input - output_layer)
        sum_sq_error =  0.5 * np.sum(error * error, axis = 0)
        avg_sum_sq_error = np.mean(sum_sq_error)
        reg_cost =  self.lambda_ * (np.sum(W1 * W1) + np.sum(W2 * W2)) / 2.0
        rho_bar = np.mean(hidden_layer, axis=1) # average activation levels 
                                                  # across hidden layer
        KL_div = np.sum(self.rho * np.log(self.rho / rho_bar) + 
                        (1 - self.rho) * np.log((1-self.rho) / (1- rho_bar)))        
        cost = avg_sum_sq_error + reg_cost + self.beta * KL_div
        
        # Back propagation
        KL_div_grad = self.beta * (- self.rho / rho_bar + (1 - self.rho) / 
                                    (1 - rho_bar))
        
        del_3 = error * output_layer * (1.0 - output_layer)
        del_2 = np.transpose(W2).dot(del_3) + KL_div_grad[:, np.newaxis]
        del_2 *= hidden_layer * (1 - hidden_layer)
        
        # Vector implementation actually calculates sum over m training 
        # examples, hence the need to divide by m         
        W1_grad = del_2.dot(visible_input.transpose()) / m
        W2_grad = del_3.dot(hidden_layer.transpose()) / m
        b1_grad = del_2
        b2_grad = del_3
        
        W1_grad += self.lambda_ * W1 # add reg term
        W2_grad += self.lambda_ * W2
        b1_grad = b1_grad.mean(axis = 1)
        b2_grad = b2_grad.mean(axis = 1)
        
        # roll out the weights and biases into single vector theta        
        theta_grad = np.concatenate((W1_grad.flatten(), W2_grad.flatten(), 
                                     b1_grad.flatten(), b2_grad.flatten()))        
        return [cost, theta_grad]

# Parameters
beta = 3.0 # sparsity parameter (rho) weight
lamda = 3e-3 # regularization weight
rho = 0.1 # sparstiy parameter i.e. target average activation for hidden  units
visible_side = 28 # sqrt of number of visible units
hidden_side = 14 # sqrt of number of hidden units
visible_size = visible_side * visible_side # number of visible units
hidden_size = hidden_side * hidden_side # number of hidden units
m = 1000     # number of training examples
max_iterations = 400 # Maximum number of iterations for numerical solver.

learning_rate = 0.0001

# data
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
training_data = mnist.train.images.T
training_data = training_data[:, 0:m]

# class
sae = sparse_autoencoder(visible_size, hidden_size, lamda, rho, beta)
current_theta = sae.initial_theta

for iter in range(max_iterations):
    cost,theta_grad = sae.cost(current_theta,training_data)
    print("Current Iter : ",iter,' Current cost: ', cost,end='\n')
    current_theta = current_theta - learning_rate * theta_grad

# -- end code --