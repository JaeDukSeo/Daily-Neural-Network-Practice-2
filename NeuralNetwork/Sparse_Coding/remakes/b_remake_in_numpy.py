import numpy as np
import numpy.linalg as la
import scipy.io
import scipy.optimize
import matplotlib.pyplot
import time
import struct
import array
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Generate training data
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(6278)
tf.set_random_seed(6728)

class sparse_autoencoder(object):

    def __init__(self, visible_size, hidden_size, lambda_, rho, beta):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.lambda_ = lambda_
        self.rho = rho
        self.beta = beta

        # initialize weights and bias terms
        w_max = np.sqrt(6.0 / (visible_size + hidden_size + 1.0))
        self.W1 = np.random.uniform(size = (visible_size, hidden_size),low=-w_max,high=w_max)
        self.W2 = np.random.uniform(size = (hidden_size, visible_size),low=-w_max,high=w_max)

        self.m1,self.v1 = np.zeros_like(self.W1),np.zeros_like(self.W1)
        self.m2,self.v2 = np.zeros_like(self.W2),np.zeros_like(self.W2)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def feedforward(self,input):
        self.hidden_layer = self.sigmoid(np.dot(input,self.W1) )
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer,self.W2) )
        self.rho_bar = np.mean(self.hidden_layer, axis=0)
        return self.output_layer,self.rho_bar

    def cost(self, visible_input):
        m = visible_input.shape[0] # number of training examples

        # Calculate the cost.
        # error = -(visible_input - self.output_layer)
        error = self.output_layer - visible_input
        sum_sq_error =  0.5 * np.sum(error * error, axis = 1)
        avg_sum_sq_error = np.mean(sum_sq_error)
        reg_cost =  self.lambda_ * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2)) / 2.0
        KL_div = self.beta * np.sum(self.rho * np.log(self.rho / self.rho_bar) +  (1 - self.rho) * np.log((1-self.rho) / (1- self.rho_bar)))
        cost = avg_sum_sq_error  + KL_div + reg_cost

        # Back propagation
        del_3 = error * self.output_layer * (1.0 - self.output_layer)
        W2_grad = self.hidden_layer.transpose().dot(del_3) / m
        W2_grad += self.lambda_ * self.W2

        self.m2 = 0.9 * self.m2 + (1.0-0.9) * W2_grad
        self.v2 = 0.999 * self.v2 + (1.0-0.999) * W2_grad ** 2
        m2_hat,v2_hat =  self.m2/(1.0-0.9),self.v2/(1.0-0.999)
        self.W2 = self.W2 - learning_rate / (np.sqrt(v2_hat) + 1e-8) * m2_hat

        KL_div_grad = self.beta * (-self.rho / self.rho_bar + (1 - self.rho) / (1- self.rho_bar) )
        del_2 = del_3.dot(self.W2.transpose())+ KL_div_grad[np.newaxis,:]
        del_2 = del_2 * self.hidden_layer * (1 - self.hidden_layer)
        W1_grad = visible_input.transpose().dot(del_2) / m
        W1_grad += self.lambda_ * self.W1

        self.m1 = 0.9 * self.m1 + (1.0-0.9) * W1_grad
        self.v1 = 0.999 * self.v1 + (1.0-0.999) * W1_grad ** 2
        m1_hat,v1_hat =  self.m1/(1.0-0.9),self.v1/(1.0-0.999)
        self.W1 = self.W1 - learning_rate / (np.sqrt(v1_hat) + 1e-8) * m1_hat

        return cost

# Parameters
beta = 3.0 # sparsity parameter (rho) weight
rho = 0.1 # sparstiy parameter i.e. target average activation for hidden units
lamda = 0.003 # regularization weight

max_iterations = 800 # Maximum number of iterations for numerical solver.
learning_rate = 0.01

visible_size = 784 # number of visible units
hidden_size = 16 # number of hidden units
m = 10000 # number of training examples
batch_size = 2000

# data
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
training_data = mnist.train.images
training_data = training_data[0:m,:]

# Create instance of autoencoder
sae = sparse_autoencoder(visible_size, hidden_size, lamda, rho, beta)
for iter in range(max_iterations):
    training_data = shuffle(training_data)
    for current_batch_index in range(0,len(training_data),batch_size):
        current_training_data = training_data[current_batch_index:current_batch_index+batch_size]
        sparse_layer,sparse_phat = sae.feedforward(current_training_data)
        cost = sae.cost(current_training_data)
        print("Current Iter : ",iter,' Current cost: ', cost,' Current Sparse: \n',np.around(sparse_phat,2))
    print('\n----------------------')

def display_network(A):
    opt_normalize = True
    opt_graycolor = True

    # Rescale
    A = A - np.average(A)

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = int(np.ceil(np.sqrt(col)))
    m = int(np.ceil(col / n))

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        image *= 0.1

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1
    fig=plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image,cmap='gray')
    plt.show()

# train data
training_data = training_data[:hidden_size,:]
training_data_reshape = np.reshape(training_data,(hidden_size,28,28))
fig=plt.figure(figsize=(10, 10))
columns = int(np.sqrt(hidden_size)); rows = int(np.sqrt(hidden_size))
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(training_data_reshape[i-1,:,:],cmap='gray')
plt.show()
plt.close('all')

# re con data
recon_data = sae.feedforward(training_data)[0]
recon_data_reshape = np.reshape(recon_data,(hidden_size,28,28))
fig=plt.figure(figsize=(10, 10))
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(recon_data_reshape[i-1,:,:],cmap='gray')
plt.show()
plt.close('all')

# Visualize the optimized activations
current_theta = sae.W1
# opt_W1 = current_theta.reshape(hidden_size, visible_size)
display_network(current_theta)
plt.close('all')

opt_W1_reshape = np.reshape(current_theta.T,(hidden_size,28,28))
fig=plt.figure(figsize=(10, 10))
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(opt_W1_reshape[i-1,:,:],cmap='gray')
plt.show()
plt.close('all')

# -- end code --
