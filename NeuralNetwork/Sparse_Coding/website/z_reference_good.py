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
   

def normalize_data(data):
    """Normalize data and rescale to fit within [0.1, 0.9] 
    
    Parameters
    ----------
    data : ndarray
           Array containing image patches
    
    Returns
    -------
    data : ndarray
           Normalized and rescaled data array
    """
    
    data = data - np.mean(data)
    pstd = 3 * np.std(data) # cutoff at 3 std dev
    
    data = np.maximum(np.minimum(data, pstd), -pstd) / pstd
    data = (data + 1.0) * 0.4 + 0.1
    return data
    

def loadMNISTImages(file_name):
    """Load and normalize MNIST images
    
    Parameters
    ----------
    file_name : str
                String with file name for image file
                 
    Returns
    -------
    data_set : ndarray
               Array of form (visible_size, number of examples)    
               
    Notes
    -----
    Written by Siddharth Agrawal for the same tutorial project.  See his github
    page.
    """
    
    # Open the file.
    image_file = open(file_name, 'rb')
    
    # Read header information from the file.    
    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)
    
    # Format the header information for useful data.    
    num_examples = struct.unpack('>I', head2)[0]
    num_rows     = struct.unpack('>I', head3)[0]
    num_cols     = struct.unpack('>I', head4)[0]
    
    # Initialize dataset as array of zeros.    
    dataset = np.zeros((num_rows*num_cols, num_examples))
    
    # Read the actual image data.    
    images_raw  = array.array('B', image_file.read())
    image_file.close()
    
    # Arrange the data in columns.    
    for i in range(num_examples):    
        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)        
        dataset[:, i] = images_raw[limit1: limit2]
    
    # Normalize and return the dataset.               
    return dataset / 255


def load_data(num_patches, patch_side):
    '''Load matlab images, randomly subsample and normalize them.
    
    Parameters
    ----------
    num_patches : int
                  Number of subsample patches to generate
    patch_side : int
                 Number of pixels per side for each subsampled patch.
                 
    Returns
    -------
    patches : ndarray
              Array of size (patch_size * patch_size, num_patches) containing
              the subsampled and normalized patches    
    '''
    
    images = scipy.io.loadmat('IMAGES.mat')
    images = images['IMAGES']
    
    patches = np.zeros((patch_side * patch_side, num_patches))
    seed = 1234 # Allow reproducible results.
    rand = np.random.RandomState(seed)
    image_index = rand.random_integers( 0, 512 - patch_side, size = 
                                        (num_patches, 2)) # upper left corner
    # There are 10 images provided:                                       
    image_number = rand.random_integers(0, 10 - 1, size = num_patches) 
    
    for i in xrange(num_patches):
        idx_1 = image_index[i, 0]
        idx_2 = image_index[i, 1]
        idx_3 = image_number[i]        
        patch = images[idx_1:idx_1 + patch_side, idx_2:idx_2 + patch_side, 
                       idx_3]
        patch = patch.flatten()        
        patches[:,i] = patch        
        
    patches = normalize_data(patches)
    
    return patches        


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
    

def simple_quadratic_function(x,y):
    """Return value and grad of f(x,y) = x^2 + 3xy
    
    Parameters
    ----------
    x : float
    y : float
    
    Returns
    -------
    val : float
          Value of function at (x,y)
    grad : ndarray
           Gradient of function in x and y directions.
    """
    
    val = x[0]**2 + 3.0 * x[0] * x[1]
    grad = np.zeros((2,))
    grad[0] = 2 * x[0] + 3.0 * x[1]
    grad[1] = 3.0 * x[0]
    
    return val, grad
    
    
def numerical_gradient(f, theta, args=None):
    """Calculate numerical gradient for function f with parameters theta.
    
    Parameters
    ----------
    f : function
        Real valued function with input parameters theta and possibly other
        required inputs taken from args in that order.
    theta : ndarray
            input parameters with which numerical derivative is calculated
            with respect to.
    args : ndarray
           remaining input parameters for function f.  No derivatives
           calculated for these.
           
    Returns
    -------
    num_grad : ndarray
               Array same length as theta containing respective partial
               derivatives.
    """
    
    eps = 1e-4 # numerical bump size
    n = theta.size
    num_grad = np.zeros(n)
    theta_pos = theta + np.eye(n) * eps
    theta_neg = theta - np.eye(n) * eps 
    
    for i, x in enumerate(theta):
        num_grad[i] = (f(theta_pos[i,:], args)[0] - f(theta_neg[i,:], args)[0]
                       ) / (2.0 * eps)
        
    return num_grad
    
def numerical_gradient_check():
    """Check analytic vs numerical derivative for autoencoder cost function 
       and simple test function.   
    """
    
    # Simple test case.
    theta =np.array([4., 10.])
    dummy = np.array([0,0])
    val, grad = simple_quadratic_function(theta, dummy)
    num_grad = numerical_gradient(simple_quadratic_function, theta)
    
    diff = la.norm(num_grad - grad) / la.norm(num_grad + grad)
    print "Simple function check: "
    print """Norm of the diff between numerical and analytic gradients: 
            {0:.4e}""".format(diff)
    
    # Parameters 
    beta = 3.0
    lambda_ = 0.0001
    rho = 0.01
    visible_side = 8
    hidden_side = 5
    visible_size = visible_side * visible_side
    hidden_size = hidden_side * hidden_side
    m = 10000 # number of training examples
    
    # Gegenerate training data 
    training_data = load_data(num_patches=m, patch_side=visible_side)
    
    sae = sparse_autoencoder(visible_size, hidden_size, lambda_, rho, beta)
    cost, grad = sae.cost(sae.initial_theta, training_data)
    num_grad = numerical_gradient(sae.cost, sae.initial_theta, training_data)
    diff = la.norm(num_grad - grad) / la.norm(num_grad + grad)
    print "Cost function check: "
    print """Norm of the diff between numerical and analytic gradients: 
            {0:.4e}""".format(diff)     
            

def run_sparse_ae():
    """Generate training data by taking random patches from 
    image files, and then use it to train a sparse autoencoder.
    Solution to 
    http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
    """
    
    # Start timing.    
    t0 = time.time()
    
    # Parameters 
    beta = 3.0 # sparsity parameter (rho) weight
    lamda = 0.0001 # regularization weight
    rho = 0.01 # sparstiy parameter i.e. target average activation for hidden 
               # units.
    visible_side = 8 # sqrt of number of visible units
    hidden_side = 5 # sqrt of number of hidden units
    visible_size = visible_side * visible_side # number of visible units
    hidden_size = hidden_side * hidden_side # number of hidden units
    m = 10000 # number of training examples
    max_iterations = 400 # Maximum number of iterations for numerical solver.
    
    # Generate training data 
    training_data = load_data(num_patches = m, patch_side = visible_side)
    # Create instance of autoencoder     
    sae = sparse_autoencoder(visible_size, hidden_size, lamda, rho, beta)
    # Train the autoencoder and retrieve optimal weights and biases
    opt_theta = sae.train(training_data, max_iterations)    
    
    # Calculate wall time
    print "Wall time: {0:.1f} seconds".format(time.time() - t0)
    
    # Visualize weights    
    opt_W1 = opt_theta[0 : visible_size * hidden_size].reshape(hidden_size, 
                                                               visible_size)    
    visualizeW1(opt_W1,visible_side, hidden_side)  
    

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
    training_data = loadMNISTImages('train-images.idx3-ubyte')  
    training_data = training_data[:, 0:m]
    
    # Create instance of autoencoder     
    sae = sparse_autoencoder(visible_size, hidden_size, lamda, rho, beta)
    # Train the autoencoder and retrieve optimal weights and biases
    opt_theta = sae.train(training_data, max_iterations)
    
    # Calculate wall time 
    print "Wall time: {0:.1f} seconds".format(time.time() - t0)
    
    # Visualize the optimized activations    
    opt_W1 = opt_theta[0 : visible_size * hidden_size].reshape(hidden_size, 
                                                               visible_size)    
    visualizeW1(opt_W1, visible_side, hidden_side)


if __name__ == "__main__":
    """Selection of programmes to run.  Uncomment the one you are interested
       in.
    """
    
#    numerical_gradient_check() 
    # run_sparse_ae() # Solution for sparse autoencoder exercise.
   run_sparse_ae_MNIST() # Solution for vectorization exercise.