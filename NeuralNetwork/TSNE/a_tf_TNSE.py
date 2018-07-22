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
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgba2rgb

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

plt.style.use('seaborn-white')
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
np.random.seed(6278)
tf.set_random_seed(6728)
ia.seed(6278)

# ======= Activation Function  ==========
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32)  + (tf_elu(tf.cast(tf.less_equal(x,0),tf.float32) * x) + 1.0)

def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1 - tf_tanh(x) ** 2

def tf_sigmoid(x): return tf.nn.sigmoid(x) 
def d_tf_sigmoid(x): return tf_sigmoid(x) * (1.0-tf_sigmoid(x))

def tf_atan(x): return tf.atan(x)
def d_tf_atan(x): return 1.0/(1.0 + x**2)

def tf_iden(x): return x
def d_tf_iden(x): return 1.0

def tf_softmax(x): return tf.nn.softmax(x)
# ======= Activation Function  ==========

# ====== miscellaneous =====
# code from: https://github.com/tensorflow/tensorflow/issues/8246
def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# ====== miscellaneous =====

# ================= VIZ =================
# Def: Simple funciton to view the histogram of weights
def show_hist_of_weigt(all_weight_list,status='before'):
    fig = plt.figure()
    weight_index = 0

    for i in range(1,1+int(len(all_weight_list)//3)):
        ax = fig.add_subplot(1,4,i)
        ax.grid(False)
        temp_weight_list = all_weight_list[weight_index:weight_index+3]
        for temp_index in range(len(temp_weight_list)):
            current_flat = temp_weight_list[temp_index].flatten()
            ax.hist(current_flat,histtype='step',bins='auto',label=str(temp_index+weight_index))
            ax.legend()
        ax.set_title('From Layer : '+str(weight_index+1)+' to '+str(weight_index+3))
        weight_index = weight_index + 3
    plt.savefig('viz/weights_'+str(status)+"_training.png")
    plt.close('all')

# Def: Simple function to show 9 image with different channels
def show_9_images(image,layer_num,image_num,channel_increase=3,alpha=None,gt=None,predict=None):
    image = (image-image.min())/(image.max()-image.min())
    fig = plt.figure()
    color_channel = 0
    limit = 10
    if alpha: limit = len(gt)
    for i in range(1,limit):
        ax = fig.add_subplot(3,3,i)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if alpha:
            ax.set_title("GT: "+str(gt[i-1])+" Predict: "+str(predict[i-1]))
        else:
            ax.set_title("Channel : " + str(color_channel) + " : " + str(color_channel+channel_increase-1))
        ax.imshow(np.squeeze(image[:,:,color_channel:color_channel+channel_increase]))
        color_channel = color_channel + channel_increase
    
    if alpha:
        plt.savefig('viz/z_'+str(alpha) + "_alpha_image.png")
    else:
        plt.savefig('viz/'+str(layer_num) + "_layer_"+str(image_num)+"_image.png")
    plt.close('all')
# ================= VIZ =================

# ================= LAYER CLASSES =================
class CNN():
    
    def __init__(self,k,inc,out,act=tf_elu,d_act=d_tf_elu):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05,seed=2))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act

    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding) 
        self.layerA = self.act(self.layer)
        return self.layerA 

    def backprop(self,gradient,stride=1,padding='SAME'):
        grad_part_1 = gradient 
        grad_part_2 = self.d_act(self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(input = grad_part_3,filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,stride,stride,1],padding=padding
        )

        grad_pass = tf.nn.conv2d_backprop_input(input_sizes = [batch_size] + list(grad_part_3.shape[1:]),filter= self.w,out_backprop = grad_middle,
            strides=[1,stride,stride,1],padding=padding
        )

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v_prev / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat)  )))         

        return grad_pass,update_w 

class CNN_Trans():
    
    def __init__(self,k,inc,out,act=tf_elu,d_act=d_tf_elu):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05,seed=2))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act

    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        output_shape2 = self.input.shape[2].value * stride
        self.layer  = tf.nn.conv2d_transpose(
            input,self.w,output_shape=[batch_size,output_shape2,output_shape2,self.w.shape[2].value],
            strides=[1,stride,stride,1],padding=padding) 
        self.layerA = self.act(self.layer)
        return self.layerA 

    def backprop(self,gradient,stride=1,padding='SAME'):
        grad_part_1 = gradient 
        grad_part_2 = self.d_act(self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(input = grad_middle,
            filter_sizes = self.w.shape,out_backprop = grad_part_3,
            strides=[1,stride,stride,1],padding=padding
        )

        grad_pass = tf.nn.conv2d(
            input=grad_middle,filter = self.w,strides=[1,stride,stride,1],padding=padding
        )
        
        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v_prev / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat)  )))         

        return grad_pass,update_w 

class FNN():
    
    def __init__(self,input_dim,hidden_dim,act,d_act):
        self.w = tf.Variable(tf.random_normal([input_dim,hidden_dim], stddev=0.05,seed=2))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act

    def feedforward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient=None):
        grad_part_1 = gradient 
        grad_part_2 = self.d_act(self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2
        grad = tf.matmul(tf.transpose(grad_part_3),grad_middle)
        grad_pass = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v_prev / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat)  )))     

        return grad_pass,update_w  

class ICA_Layer():

    def __init__(self,inc):
        self.w_ica = tf.Variable(tf.random_normal([inc,inc],stddev=0.05,seed=2)) 
        # self.w_ica = tf.Variable(tf.eye(inc)*0.0001) 

    def feedforward(self,input):
        self.input = input
        self.ica_est = tf.matmul(input,self.w_ica)
        self.ica_est_act = tf_atan(self.ica_est)
        return self.ica_est_act

    def backprop(self):
        grad_part_2 = d_tf_atan(self.ica_est)
        grad_part_3 = self.input

        grad_pass = tf.matmul(grad_part_2,tf.transpose(self.w_ica))
        g_tf = tf.linalg.inv(tf.transpose(self.w_ica)) - (2/batch_size) * tf.matmul(tf.transpose(self.input),self.ica_est_act)

        update_w = []
        update_w.append(tf.assign(self.w_ica,self.w_ica+0.2*g_tf))

        return grad_pass,update_w  

# ================= LAYER CLASSES =================

# data
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
x_data, train_label, y_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# x_data = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
# y_data = y_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
train_batch = np.zeros((55000,28,28,1))
test_batch = np.zeros((10000,28,28,1))

# for x in range(len(x_data)):
#     train_batch[x,:,:,:] = np.expand_dims(imresize(x_data[x,:,:,0],(28,28)),axis=3)
# for x in range(len(y_data)):
#     test_batch[x,:,:,:] = np.expand_dims(imresize(y_data[x,:,:,0],(28,28)),axis=3)

# simple normalize
train_batch = train_batch/255.0
test_batch = test_batch/255.0

train_batch = x_data[:3]
train_label = train_label[:3]
test_batch = y_data[:200]
test_label = test_label[:200]

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(train_batch.max(),train_batch.min())

print(test_batch.shape)
print(test_label.shape)
print(test_batch.max(),test_batch.min())

# ======= TSNE ======
def neg_distance(X):
    X_sum = np.sum(X**2,1)
    distance = np.reshape(X_sum,[-1,1])
    return -(distance - 2*X.dot(X.T)+distance.T) 

def softmax_max(X,diag=True):
    X_exp = np.exp(X - X.max(1).reshape([-1, 1]))
    X_exp = X_exp + 1e-10
    if diag: np.fill_diagonal(X_exp, 0.)
    # X_exp = X_exp + 1e-10
    return X_exp/X_exp.sum(1).reshape([-1, 1])

def calc_prob_matrix(distances, sigmas=None):
    """Convert a distances matrix to a matrix of probabilities."""
    if sigmas is not None:
        two_sig_sq = 2. * sigmas.reshape([-1, 1]) ** 2
        return softmax_max(distances / two_sig_sq)
    else:
        return softmax_max(distances)

def perplexity(distances, sigmas):
    """Wrapper function for quick calculation of 
    perplexity over a distance matrix."""
    prob_matrix = calc_prob_matrix(distances, sigmas)
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix + 1e-10), 1)
    perplexity = 2.0 ** entropy
    return perplexity

def binary_search(distance_vec, target, max_iter=20000,tol=1e-13, lower=1e-10, upper=1e10):
    """Perform a binary search over input values to eval_fn.
    # Arguments
        eval_fn: Function that we are optimising over.
        target: Target value we want the function to output.
        tol: Float, once our guess is this close to target, stop.
        max_iter: Integer, maximum num. iterations to search for.
        lower: Float, lower bound of search range.
        upper: Float, upper bound of search range.
    # Returns:
        Float, best input value to function found during search.
    """
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = perplexity(distance_vec,np.array(guess))
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess

def find_optimal_sigmas(distances, target_perplexity):
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role."""
    sigmas = [] 
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(distances[i:i+1, :], target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)

def p_conditional_to_joint(P):
    """Given conditional probabilities matrix P, return
    approximation of joint distribution probabilities."""
    return (P + P.T) / (2. * P.shape[0])

def p_joint(X, target_perplexity):
    """Given a data matrix X, gives joint probabilities matrix.

    # Arguments
        X: Input data matrix.
    # Returns:
        P: Matrix with entries p_ij = joint probabilities.
    """
    # Get the negative euclidian distances matrix for our data
    distances = neg_distance(X)
    # Find optimal sigma for each row of this distances matrix
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(distances, sigmas)
    # Go from conditional to joint probabilities matrix
    P = p_conditional_to_joint(p_conditional)
    return P
# ======= TSNE ======

# ====== TF TSNE ====
def tf_neg_distance(X):
    X_sum = tf.reduce_sum(X**2,1)
    distance = tf.reshape(X_sum,[-1,1])
    return -(distance - 2*tf.matmul(X,tf.transpose(X))+tf.transpose(distance)) 

def tf_q_tsne(Y):
    distances = tf_neg_distance(Y)
    inv_distances = tf.pow(1. - distances, -1)
    inv_distances = tf.matrix_set_diag(inv_distances,tf.zeros([inv_distances.shape[0].value],dtype=tf.float32)) 
    return inv_distances / tf.reduce_sum(inv_distances), inv_distances


# ====== TF TSNE ====


# hyper
perplexity_number = 20
reduced_dimension = 2
P = p_joint(train_batch,perplexity_number)

# graph
W = tf.Variable(tf.random_normal(shape=[P.shape[0],reduced_dimension],dtype=tf.float32,stddev=0.05))

one = tf_q_tsne(W)


# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    sess_results = sess.run(one)
    print(sess_results[0].shape)
    print(sess_results[0])
    print(sess_results[1].shape)
    print(sess_results[1])
# -- end code --