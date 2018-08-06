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

import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class sparse_autoencoder(object):
    """
    Creates an instance of a sparse autoencoder.
    
    Attributes
    ----------
    visible_size : int
                   Number of input units
    hidden_size : int
                  Number of hidden units
    lambda_ : float
              Weight decay parameter
    rho : float
          Target average activation for the hidden units
    beta : float
           Weighting on the sparsity penalty term
    idx_0 : int
            Starting index for weight W1 terms in theta array.  Should be 0.
    idx_1 : int
            Start index for weight W2 terms in theta array.  
    idx_2 : int 
            Starting index for bias b1 terms in theta array.
    idx_3 : int
            Starting index for bias b2 terms in theta array.
    initial_theta : ndarray
                    Array with appropriately generated random entries.  
                    Used as initial guess in numerical calculation.              
    """
    
    def __init__(self, visible_size, hidden_size, lambda_, rho, beta):
        """Create instance of sparse autoencoder
        
        Parameters
        ----------
        visible_size : int
                       the number of input units (probably 64) 
        hidden_size : int 
                      the number of hidden units (probably 25) 
        lambda_ : float
                weight decay parameter
        rho : float
              The desired average activation for the hidden units 
        beta : float
               weight of sparsity penalty term             
        """   
        
        self.visible_size = visible_size  
        self.hidden_size = hidden_size 
        self.lambda_ = lambda_ 
        self.rho = rho 
        self.beta = beta 
                
        # initialize weights and bias terms 
        w_max = np.sqrt(6.0 / (visible_size + hidden_size + 1.0))
        w_min = -w_max
        W1 = (w_max - w_min) * np.random.random_sample(size = (hidden_size, 
                                                        visible_size)) + w_min
        W2 = (w_max - w_min) * np.random.random_sample(size = (visible_size, 
                                                        hidden_size)) + w_min
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
        """ Apply sigmoid transorm to array
        
        Parameters
        ----------
        x : array_like
            input array
            
        Returns
        -------
        ndarray
            Array with element-wise transform applied
        """
        
        return 1.0 / (1.0 + np.exp(-x))
    
    def unpack_theta(self, theta):
        """Break up theta array into  2 weight and 2 bias arrays
        
        Parameters
        ----------
        theta : ndarray
                Array containing weights (W1, W2), and biases (b1 and b2) in 
                that order.
                
        Returns
        -------
        W1, W2, b1, b2 : ndarrays
                         W1 of form (hidden_size, visible_size), W2 of form 
                         (visible_size, hidden_size), b1 of form (hidden_size,
                         1) and b2 of form (visible_size, 1)
        """
                
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
        """Evaluate sigmoidal cost function for given theta and input array
        
        Parameters
        ----------
        theta : ndarray
                vector containing current weights and biases
        visible_input : ndarray
                        2d vector number_visible_units x num_training_examples
                        
        Returns
        -------
        [cost, theta_grad] : array 
                             cost is sum of squares errors plus KL penalty term
                             theta_grad is type ndarray and contains analytic
                             partial derivative wrt each weight and bias.
        """
        
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
        
        
    def train(self, data, max_iterations):
        """ Return optimal theta for a given set of data 
        
        Parameters
        ----------
        data : ndarray
               Array of form (visible_size, number of patches)       
        max_iterations : int
                         Maximum number of iterations to use in the numerical
                         solver
                         
        Returns
        -------
        opt_theta : ndarray
                    Array containing weights and biases which minimize the
                    cost function.
        """
        
        opt_soln = scipy.optimize.minimize(self.cost, 
                                           self.initial_theta, 
                                           args = (data,), method = 'L-BFGS-B',
                                           jac = True, options = 
                                           {'maxiter':max_iterations} )
        opt_theta = opt_soln.x
        return opt_theta

def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):
    """Visualize the obtained W1 values as images
    
    Parameters
    ----------
    opt_W1 : ndarray
             Array of dimension (hidden_size, visible_size) containing weights
             between visible input and hidden layer.
    vis_patch_side : int
                     Number of pixels per side of images to be displayed
    hid_patch_side : int
                     Subplot of size (hid_patch_side, hid_patch_side) images
                     to be displayed. 
                     
    Notes
    -----
    Written by Siddharth Agrawal for the same tutorial project.  See his github
    page.    
    """
    
    figure, axes = matplotlib.pyplot.subplots(nrows = hid_patch_side,
                                              ncols = hid_patch_side)
    index = 0                                              
    for axis in axes.flat:    
        # Add row of weights as an image to the plot.    
        image = axis.imshow(opt_W1[index, :].reshape(vis_patch_side, 
                            vis_patch_side), cmap = matplotlib.pyplot.cm.gray,
                            interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1
        
    # Show the obtained plot.        
    matplotlib.pyplot.show()        

def run_sparse_ae_MNIST():
    """  We generate training data by taking random patches from MNIST 
    image files, and then use it to train a sparse autoencoder
    Solution to 
    http://ufldl.stanford.edu/wiki/index.php/Exercise:Vectorization
    """
    
    # Start timing    
    t0 = time.time()
    
    # Parameters
    beta = 3.0 # sparsity parameter (rho) weight
    lamda = 3e-3 # regularization weight
    rho = 0.1 # sparstiy parameter i.e. target average activation for hidden 
              # units
    visible_side = 28 # sqrt of number of visible units
    hidden_side = 14 # sqrt of number of hidden units
    visible_size = visible_side * visible_side # number of visible units
    hidden_size = hidden_side * hidden_side # number of hidden units
    m = 10000 # number of training examples
    max_iterations = 400 # Maximum number of iterations for numerical solver.
    
    # Generate training data 
    from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist = input_data.read_data_sets('../../../Dataset/MNIST/', one_hot=True)
    training_data = mnist.train.images.T
    # training_data = loadMNISTImages('train-images.idx3-ubyte')  
    training_data = training_data[:, 0:m]
    print(training_data.shape)
    
    # Create instance of autoencoder     
    sae = sparse_autoencoder(visible_size, hidden_size, lamda, rho, beta)
    # Train the autoencoder and retrieve optimal weights and biases
    opt_theta = sae.train(training_data, max_iterations)
    
    # Calculate wall time 
    print("Wall time: {0:.1f} seconds".format(time.time() - t0))
    
    # Visualize the optimized activations    
    opt_W1 = opt_theta[0 : visible_size * hidden_size].reshape(hidden_size, visible_size)    
    visualizeW1(opt_W1, visible_side, hidden_side)


if __name__ == "__main__":
    """Selection of programmes to run.  Uncomment the one you are interested
       in.
    """
    run_sparse_ae_MNIST() # Solution for vectorization exercise.


# -- end code --