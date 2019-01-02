# ========= choice of library ====
import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread,imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from imgaug import augmenters as iaa
import nibabel as nib
import imgaug as ia
from scipy.ndimage import zoom
from sklearn.utils import shuffle
import matplotlib.animation as animation

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(6278)
tf.set_random_seed(6728)
ia.seed(6278)

# Generate training data
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
# ========= choice of library ====

# ======= Activation Function  ==========
def tf_elu(x):
    """ Exponential Linear Unit based on the ICCV 2015 paper
    https://arxiv.org/pdf/1511.07289.pdf

    Parameters
    ----------
    x : float
        The floating point number that is going to be applied to the ELU activation.

    Returns
    -------
    float
        Data with the same dimensions as the input after ELU

    """
    return tf.nn.elu(x)
def d_tf_elu(x):
    """Derivative of the Exponential Linear Unit base on the ICCV 2015 paper
    https://arxiv.org/pdf/1511.07289.pdf

    Parameters
    ----------
    x : type
        Description of parameter `x`.

    Returns
    -------
    type
        Description of returned object.

    """

    return tf.cast(tf.greater(x,0),tf.float64)  + (tf_elu(tf.cast(tf.less_equal(x,0),tf.float64) * x) + 1.0)

def tf_relu(x):   return tf.nn.relu(x)
def d_tf_relu(x): return tf.cast(tf.greater(x,0),tf.float64)

def tf_tanh(x):   return tf.nn.tanh(x)
def d_tf_tanh(x): return 1 - tf_tanh(x) ** 2

def tf_sigmoid(x):   return tf.nn.sigmoid(x)
def d_tf_sigmoid(x): return tf_sigmoid(x) * (1.0-tf_sigmoid(x))

def tf_atan(x):   return tf.atan(x)
def d_tf_atan(x): return 1.0/(1.0 + x**2)

def tf_iden(x):   return x
def d_tf_iden(x): return x

def tf_softmax(x): return tf.nn.softmax(x)
def softabs(x):    return tf.sqrt(x ** 2 + 1e-20)

def tf_logcosh(x):   return tf.log(tf.cosh(x))
def d_tf_logcosh(x): return tf.tanh(x)

# ---- FAST ICA Special function ----
def tf_cube(x):   return x ** 3
def d_tf_cube(x): return 2* x ** 2
def tf_exp(x):    return x * tf.exp(-(x**2)/2.0)
def d_tf_exp(x):  return (1-x**2) * tf.exp(-(x**2)/2.0)
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
    """Unpickles the data from the given path.

    Parameters
    ----------
    file : string
        location of the pickle file to unpickle

    Returns
    -------
    dict
        Dictionary of the unpickled data.

    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# Func: Display Weights (Soruce: https://github.com/jonsondag/ufldl_templates/blob/master/display_network.py)
def display_network(A,current_iter=None):
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
    plt.axis('off')
    plt.tight_layout()
    return [plt.imshow(image,cmap='gray', animated=True),plt.text(0.5, 1.0, 'Current Iter : '+str(current_iter),color='red', fontsize=30,horizontalalignment='center', verticalalignment='top')]
# ====== miscellaneous =====

# ================= LAYER CLASSES =================
class CNN():

    def __init__(self,k,inc,out, stddev=0.05,which_reg=0,act=tf_tanh,d_act=d_tf_tanh):
        self.w          = tf.Variable(tf.random_normal([k,k,inc,out],stddev=stddev,seed=2,dtype=tf.float64))
        self.b          = tf.Variable(tf.zeros(out,dtype=tf.float64))
        self.m,self.v   = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.mb,self.vb = tf.Variable(tf.zeros_like(self.b)),tf.Variable(tf.zeros_like(self.b))
        self.act,self.d_act = act,d_act
        self.which_reg  = which_reg
        
    def getw(self): return [self.w,self.b]

    def feedforward(self,input,stride=1,padding='VALID'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding) + self.b 
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient,stride=1,padding='VALID'):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2
        grad_b      = tf.reduce_mean(grad_middle,(0,1,2))/batch_size
        grad        = tf.nn.conv2d_backprop_filter(input = grad_part_3,filter_sizes = tf.shape(self.w),  out_backprop = grad_middle,strides=[1,stride,stride,1],padding=padding) / batch_size
        grad_pass   = tf.nn.conv2d_backprop_input (input_sizes = tf.shape(self.input),filter= self.w,out_backprop = grad_middle,strides=[1,stride,stride,1],padding=padding)

        if self.which_reg == 0:   grad = grad
        if self.which_reg == 0.5: grad = grad + lamda * (tf.sqrt(tf.abs(self.w))) * (1.0/tf.sqrt(tf.abs(self.w)+ 10e-5)) * tf.sign(self.w)
        if self.which_reg == 1:   grad = grad + lamda * tf.sign(self.w)
        if self.which_reg == 1.5: grad = grad + lamda * 1.0/(tf.sqrt(tf.square(self.w) + 10e-5)) * self.w
        if self.which_reg == 2:   grad = grad + lamda * (1.0/tf.sqrt(tf.square(tf.abs(self.w))+ 10e-5)) * tf.abs(self.w) * tf.sign(self.w)
        if self.which_reg == 2.5: grad = grad + lamda * 2.0 * self.w
        if self.which_reg == 3:   grad = grad + lamda * tf.pow(tf.pow(tf.abs(self.w),3)+ 10e-5,-0.66) * tf.pow(tf.abs(self.w),2) * tf.sign(self.w)
        if self.which_reg == 4:   grad = grad + lamda * tf.pow(tf.pow(tf.abs(self.w),4)+ 10e-5,-0.75) * tf.pow(tf.abs(self.w),3) * tf.sign(self.w)

        update_w = []
        
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v,self.v*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1) ; v_hat = self.v / (1-beta2)
        adam_middle = m_hat * learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middle  )))
        
        update_w.append(tf.assign( self.mb,self.mb*beta1 + (1-beta1) * (grad_b)   ))
        update_w.append(tf.assign( self.vb,self.vb*beta2 + (1-beta2) * (grad_b ** 2)   ))
        m_hatb = self.mb / (1-beta1) ; v_hatb = self.vb / (1-beta2)
        adam_middleb = m_hatb * learning_rate/(tf.sqrt(v_hatb) + adam_e)
        update_w.append(tf.assign(self.b,tf.subtract(self.b,adam_middleb  )))
        
        return grad_pass,update_w

# Func: 3D Convolutional Layer
class CNN_3D():

    def __init__(self,filter_depth,filter_height,filter_width,in_channels,out_channels,act=tf_elu,d_act=d_tf_elu):
        self.w = tf.Variable(tf.random_normal([filter_depth,filter_height,filter_width,in_channels,out_channels],stddev=0.05,seed=2,dtype=tf.float64))
        self.b = tf.Variable(tf.random_normal([out_channels],stddev=0.05,seed=2,dtype=tf.float64))
        self.act,self.d_act = act,d_act
    def getw(self): return self.w
    def feedforward(self,input,stride=1,padding='SAME',res=True):
        self.input  = input
        self.layer  = tf.nn.conv3d(input,self.w,strides=[1,1,1,1,1],padding=padding) + self.b
        self.layerA = self.act(self.layer)
        if res:  return self.layerA + self.input
        else:    return self.layerA

    def backprop(self): raise NotImplementedError("Not Implemented Yet")

# Func: Transpose Convolutional Layer
class CNN_Trans():

    def __init__(self,k,inc,out,act=tf_elu,d_act=d_tf_elu):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05,seed=2,dtype=tf.float64))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act

    def getw(self): return self.w
    def feedforward(self,input,output_shape,stride=1,padding='SAME'):
        self.input   = input
        self.layer  = tf.nn.conv2d_transpose( input,self.w,output_shape=[batch_size,output_shape,output_shape,self.w.shape[2].value],
            strides=[1,stride,stride,1],padding=padding)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient,stride_grad=1,stride=1,padding='SAME',which_reg=0):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input
        grad_middle = grad_part_1 * grad_part_2
        grad = tf.nn.conv2d_backprop_filter(input = grad_middle,filter_sizes = self.w.shape,out_backprop = grad_part_3,
            strides=[1,stride_grad,stride_grad,1],padding=padding
        ) / batch_size

        grad_pass = tf.nn.conv2d(input=grad_middle,filter = self.w,strides=[1,stride,stride,1],padding=padding
        )

        if which_reg == 0:    grad = grad
        if which_reg == 0.5:  grad = grad + lamda * (tf.sqrt(tf.abs(self.w))) * (1.0/tf.sqrt(tf.abs(self.w)+ 10e-5)) * tf.sign(self.w)
        if which_reg == 1:    grad = grad + lamda * tf.sign(self.w)
        if which_reg == 1.5:  grad = grad + lamda * 1.0/(tf.sqrt(tf.square(self.w) + 10e-5)) * self.w
        if which_reg == 2:    grad = grad + lamda * (1.0/tf.sqrt(tf.square(tf.abs(self.w))+ 10e-5)) * tf.abs(self.w) * tf.sign(self.w)
        if which_reg == 2.5:  grad = grad + lamda * 2.0 * self.w
        if which_reg == 3:    grad = grad + lamda * tf.pow(tf.pow(tf.abs(self.w),3)+ 10e-5,-0.66) * tf.pow(tf.abs(self.w),2) * tf.sign(self.w)
        if which_reg == 4:    grad = grad + lamda * tf.pow(tf.pow(tf.abs(self.w),4)+ 10e-5,-0.75) * tf.pow(tf.abs(self.w),3) * tf.sign(self.w)

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v,self.v*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat)  )))

        return grad_pass,update_w

# Func: Recurrent Convolutional Layer
class RNN_CNN():

    def __init__(self,timestamp,c_in,c_out,x_kernel,h_kernel,size,act=tf_elu,d_act=d_tf_elu):

        self.w = tf.Variable(tf.random_normal([x_kernel,x_kernel,c_in,c_out],stddev=0.05,seed=2,dtype=tf.float64))
        self.h = tf.Variable(tf.random_normal([h_kernel,h_kernel,c_out,c_out],stddev=0.05,seed=2,dtype=tf.float64))

        self.act = act; self.d_act = d_act

        self.input_record   = tf.Variable(tf.zeros([timestamp,batch_size,size,size,c_in],tf.float64))
        self.hidden_record  = tf.Variable(tf.zeros([timestamp+1,batch_size,size,size,c_out],tf.float64))
        self.hiddenA_record = tf.Variable(tf.zeros([timestamp+1,batch_size,size,size,c_out],tf.float64))

        self.m_x,self.v_x = tf.Variable(tf.zeros_like(self.w,dtype=tf.float64)),tf.Variable(tf.zeros_like(self.w,dtype=tf.float64))
        self.m_h,self.v_h = tf.Variable(tf.zeros_like(self.h,dtype=tf.float64)),tf.Variable(tf.zeros_like(self.h,dtype=tf.float64))

    def feedforward(self,input,timestamp):

        # assign the input for back prop
        hidden_assign = []
        hidden_assign.append(tf.assign(self.input_record[timestamp,:,:,:],input))

        # perform feed forward
        layer =  tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')  + tf.nn.conv2d(self.hidden_record[timestamp,:,:,:,:],self.h,strides=[1,1,1,1],padding='SAME')
        layerA = self.act(layer)

        # assign for back prop
        hidden_assign.append(tf.assign(self.hidden_record[timestamp+1,:,:,:,:],layer))
        hidden_assign.append(tf.assign(self.hiddenA_record[timestamp+1,:,:,:,:],layerA))

        return layerA, hidden_assign

    def backprop(self,grad,timestamp):

        grad_1 = grad
        grad_2 = self.d_act(self.hidden_record[timestamp,:,:,:,:])
        grad_3_x = self.input_record[timestamp,:,:,:,:]
        grad_3_h = self.hiddenA_record[timestamp-1,:,:,:,:]
        grad_middle = grad_1 * grad_2

        grad_x = tf.nn.conv2d_backprop_filter(input=grad_3_x,filter_size = self.w.shape,out_backprop = grad_middle,strides=[1,1,1,1],padding='SAME')
        grad_h = tf.nn.conv2d_backprop_filter(
            input=grad_3_h,filter_size = self.h.shape,
            out_backprop = grad_middle,strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_size = self.hiddenA_record[timestamp-1,:,:,:].shape,
            filter=self.h,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        update_w = []
        # === update x ====
        update_w.append( tf.assign(self.m_x,beta_1*self.m_x + (1-beta_1) * grad_x)  )
        update_w.append( tf.assign(self.v_x,beta_2*self.v_x + (1-beta_2) * grad_x ** 2) )
        m_hat_x = self.m_x/(1-beta_1)
        v_hat_x = self.v_x/(1-beta_2)
        adam_middle_x = learning_rate/(tf.sqrt(v_hat_x) + adam_e)
        update_w.append( tf.assign(self.w_x, tf.subtract(self.w_x,adam_middle_x*m_hat_x))  )

        # === update h ====
        update_w.append( tf.assign(self.m_h,beta_1*self.m_h + (1-beta_1) * grad_h)  )
        update_w.append( tf.assign(self.v_h,beta_2*self.v_h + (1-beta_2) * grad_h ** 2) )
        m_hat_h = self.m_h/(1-beta_1)
        v_hat_h = self.v_h/(1-beta_2)
        adam_middle_h = learning_rate/(tf.sqrt(v_hat_h) + adam_e)
        update_w.append( tf.assign(self.w_h, tf.subtract(self.w_h,adam_middle_h*m_hat_h))  )

        return grad_pass,update_w

# Func: ZigZag RNN CNN
class ZigZag_RNN_CNN():

    def __init__(self,timestamp,c_in,c_out,x_kernel,h_kernel,size,act=tf_elu,d_act=d_tf_elu):

        self.w_1 = tf.Variable(tf.random_normal([x_kernel,x_kernel,c_in,c_out],stddev=0.05,seed=2))
        self.h_1 = tf.Variable(tf.random_normal([h_kernel,h_kernel,c_out,c_out],stddev=0.05,seed=2))

        self.act = act; self.d_act = d_act

        self.w_2 = tf.Variable(tf.random_normal([x_kernel,x_kernel,c_in,c_out],stddev=0.05,seed=2))
        self.h_2 = tf.Variable(tf.random_normal([h_kernel,h_kernel,c_out,c_out],stddev=0.05,seed=2))

        self.input_record_1   = tf.Variable(tf.zeros([timestamp,batch_size//2,size,size,c_in]))
        self.hidden_record_1  = tf.Variable(tf.zeros([timestamp+1,batch_size//2,size,size,c_out]))
        self.hiddenA_record_1 = tf.Variable(tf.zeros([timestamp+1,batch_size//2,size,size,c_out]))

        self.input_record_2   = tf.Variable(tf.zeros([timestamp,batch_size//2,size,size,c_in]))
        self.hidden_record_2  = tf.Variable(tf.zeros([timestamp+1,batch_size//2,size,size,c_out]))
        self.hiddenA_record_2 = tf.Variable(tf.zeros([timestamp+1,batch_size//2,size,size,c_out]))

    def feedforward_straight(self,input1,input2,timestamp):

        # assign the inputs
        hidden_assign = []

        # perform feed forward on left
        layer_1 =  tf.nn.conv2d(input1,self.w_1,strides=[1,1,1,1],padding='SAME')  + \
        tf.nn.conv2d(self.hiddenA_record_1[timestamp,:,:,:,:],self.h_1,strides=[1,1,1,1],padding='SAME')
        layerA_1 = self.act(layer_1)

        # perform feed forward on right
        layer_2 =  tf.nn.conv2d(input2,self.w_2,strides=[1,1,1,1],padding='SAME')  + \
        tf.nn.conv2d(self.hiddenA_record_2[timestamp,:,:,:,:],self.h_2,strides=[1,1,1,1],padding='SAME')
        layerA_2 = self.act(layer_2)

        # assign for left
        hidden_assign.append(tf.assign(self.hidden_record_1[timestamp+1,:,:,:,:],layer_1))
        hidden_assign.append(tf.assign(self.hiddenA_record_1[timestamp+1,:,:,:,:],layerA_1))

        # assign for right
        hidden_assign.append(tf.assign(self.hidden_record_2[timestamp+1,:,:,:,:],layer_2))
        hidden_assign.append(tf.assign(self.hiddenA_record_2[timestamp+1,:,:,:,:],layerA_2))

        return layerA_1,layerA_2,hidden_assign

    def feedforward_zigzag(self,input1,input2,timestamp):

        # assign the inputs
        hidden_assign = []

        # perform feed forward on left
        layer_1 =  tf.nn.conv2d(input1,self.w_1,strides=[1,1,1,1],padding='SAME')  + \
        tf.nn.conv2d(self.hiddenA_record_2[timestamp,:,:,:,:],self.h_1,strides=[1,1,1,1],padding='SAME')
        layerA_1 = self.d_act(layer_1)

        # perform feed forward on right
        layer_2 =  tf.nn.conv2d(input2,self.w_2,strides=[1,1,1,1],padding='SAME')  + \
        tf.nn.conv2d(self.hiddenA_record_1[timestamp,:,:,:,:],self.h_2,strides=[1,1,1,1],padding='SAME')
        layerA_2 = self.d_act(layer_2)

        # assign for left
        hidden_assign.append(tf.assign(self.hidden_record_1[timestamp+1,:,:,:,:],layer_1))
        hidden_assign.append(tf.assign(self.hiddenA_record_1[timestamp+1,:,:,:,:],layerA_1))

        # assign for right
        hidden_assign.append(tf.assign(self.hidden_record_2[timestamp+1,:,:,:,:],layer_2))
        hidden_assign.append(tf.assign(self.hiddenA_record_2[timestamp+1,:,:,:,:],layerA_2))

        return layerA_1,layerA_2,hidden_assign

# Func: LSTM CNN
class LSTM_CNN():

    def __init__(self):
        raise NotImplementedError("Not Implemented Yet")

# Func: Fully Connected Layer
class FNN():

    def __init__(self,inc,outc,act=tf_elu,d_act=d_tf_elu,special_init=False,which_reg=0.0):
        if special_init:
            interval = np.sqrt(6.0 / (inc + outc + 1.0))
            self.w = tf.Variable(tf.random_uniform(shape=(inc, outc),minval=-interval,maxval=interval,dtype=tf.float32,seed=2))
            self.b = tf.Variable(tf.random_uniform(shape=(outc),minval=-interval,maxval=interval,dtype=tf.float32,seed=2))
        else:
            self.w = tf.Variable(tf.random_normal([inc,outc], stddev=0.05,seed=2,dtype=tf.float32))
            self.b = tf.Variable(tf.random_normal([outc], stddev=0.05,seed=2,dtype=tf.float32))

        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.m_b,self.v_b = tf.Variable(tf.zeros_like(self.b)),tf.Variable(tf.zeros_like(self.b))
        self.act,self.d_act = act,d_act
        self.which_reg = which_reg

    def getw(self): return self.w

    def feedforward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w) + self.b
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient=None,which_reg=0):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2
        grad  = tf.matmul(tf.transpose(grad_part_3),grad_middle)/batch_size
        grad_b= tf.reduce_mean(grad_middle,axis=0)
        grad_pass = tf.matmul(grad_middle,tf.transpose(self.w))

        # === Reg ===
        if self.which_reg == 0:
            grad  = grad
            grad_b= grad_b

        if self.which_reg == 0.5:
            grad  = grad + lamda * (tf.sqrt(tf.abs(self.w))) * (1.0/tf.sqrt(tf.abs(self.w)+ 10e-5)) * tf.sign(self.w)
            grad_b= grad_b+lamda * (tf.sqrt(tf.abs(self.b))) * (1.0/tf.sqrt(tf.abs(self.b)+ 10e-5)) * tf.sign(self.b)

        if self.which_reg == 1:
            grad = grad   + lamda * tf.sign(self.w)
            grad_b=grad_b + lamda * tf.sign(self.b)

        if self.which_reg == 1.5:
            grad = grad   + lamda * 1.0/(tf.sqrt(tf.square(self.w) + 10e-5)) * self.w
            grad_b=grad_b + lamda * 1.0/(tf.sqrt(tf.square(self.b) + 10e-5)) * self.b

        if self.which_reg == 2:
            grad = grad  + lamda * (1.0/tf.sqrt(tf.square(tf.abs(self.w))+ 10e-5)) * tf.abs(self.w) * tf.sign(self.w)
            grad_b=grad_b+ lamda * (1.0/tf.sqrt(tf.square(tf.abs(self.b))+ 10e-5)) * tf.abs(self.b) * tf.sign(self.b)

        if self.which_reg == 2.5:
            grad = grad   + lamda * 2.0 * self.w
            grad_b=grad_b + lamda * 2.0 * self.b

        if self.which_reg == 3:
            grad = grad   + lamda * tf.pow(tf.pow(tf.abs(self.w),3)+ 10e-5,-0.66) * tf.pow(tf.abs(self.w),2) * tf.sign(self.w)
            grad_b=grad_b + lamda * tf.pow(tf.pow(tf.abs(self.b),3)+ 10e-5,-0.66) * tf.pow(tf.abs(self.b),2) * tf.sign(self.b)

        if self.which_reg == 4:
            grad = grad   + lamda * tf.pow(tf.pow(tf.abs(self.w),4)+ 10e-5,-0.75) * tf.pow(tf.abs(self.w),3) * tf.sign(self.w)
            grad_b=grad_b + lamda * tf.pow(tf.pow(tf.abs(self.b),4)+ 10e-5,-0.75) * tf.pow(tf.abs(self.b),3) * tf.sign(self.b)

        update_w = []

        # Update the Weight First
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v,self.v*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middle = m_hat *  learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middle )))

        # Update the Bias later
        update_w.append(tf.assign(self.m_b,self.m_b*beta1 + (1-beta1) * (grad_b)   ))
        update_w.append(tf.assign(self.v_b,self.v_b*beta2 + (1-beta2) * (grad_b ** 2)   ))
        m_hat_b = self.m_b / (1-beta1)
        v_hat_b = self.v_b / (1-beta2)
        adam_middle_b = m_hat_b *  learning_rate/(tf.sqrt(v_hat_b) + adam_e)
        update_w.append(tf.assign(self.b,tf.subtract(self.b,adam_middle_b )))

        return grad_pass,update_w


# Func: Fully Connected RNN Layer
class RNN():

    def __init__(self):
        raise NotImplementedError("Not Implemented Yet")

# Func: Fully Connnected LSTM Layer
class LSTM():

    def __init__(self):
        raise NotImplementedError("Not Implemented Yet")

# Func: Layer for Sparse Coding
class sparse_code_layer():

    def __init__(self,inc,outc,sparsity=0.1,special_init=False,act=tf_sigmoid,d_act=d_tf_sigmoid):

        if special_init:
            interval = np.sqrt(6.0 / (inc + outc + 1.0))
            self.w  = tf.Variable(tf.random_uniform(shape=(inc, outc),minval=-interval,maxval=interval,dtype=tf.float64,seed=4))
        else:
            self.w = tf.Variable(tf.random_normal([inc,outc], stddev=0.05,seed=2,dtype=tf.float64))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.act = act ; self.d_act = d_act

    def getw(self): return self.w

    def feedforward(self,input):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = self.act(self.layer)
        self.current_sparsity = tf.reduce_mean(self.layerA, axis=0)
        return self.layerA,self.current_sparsity

    def backprop(self,gradient,l2_regularization=True):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input
        grad_part_KL = beta * (- aimed_sparsity / self.current_sparsity + (1.0 - aimed_sparsity) / (1.0 - self.current_sparsity))
        grad_part_1 = grad_part_1 + grad_part_KL[tf.newaxis,:]

        grad_middle = grad_part_1 * grad_part_2
        grad = tf.matmul(tf.transpose(grad_part_3),grad_middle)/batch_size
        grad_pass = tf.matmul(grad_middle,tf.transpose(self.w))

        if l2_regularization:
            grad = grad + lamda * self.w

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v,self.v*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middle = learning_rate/(tf.sqrt(v_hat) + adam_e) * m_hat
        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middle )))

        return grad_pass,update_w

# Func: Simple Sparse
class simple_sparse_layer():

    def __init__(self,inc,outc,special_init=False):
        if special_init:
            interval = np.sqrt(6.0 / (inc + outc + 1.0))
            self.w  = tf.Variable(tf.random_uniform(shape=(inc, outc),minval=-interval,maxval=interval,dtype=tf.float64,seed=4))
        else:
            self.w = tf.Variable(tf.random_normal([inc,outc], stddev=0.05,seed=2,dtype=tf.float64))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,top_size=1):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.top = tf.nn.top_k(self.layer, top_size)
        self.top_mean = tf.reduce_mean(self.top.values)
        self.top_mean_mask = tf.cast(tf.greater_equal(self.layer, self.top_mean),tf.float64)
        self.x_hat = self.top_mean_mask * self.layer
        self.reconstructed_layer = tf.matmul(self.x_hat,tf.transpose(self.w))
        return self.reconstructed_layer

    def backprop(self,l2_regularization=False):
        w_update_1 = tf.expand_dims(tf.reduce_sum(self.input - self.reconstructed_layer,axis=0),0)
        w_update_2 = tf.expand_dims(tf.reduce_sum(tf.sign(self.x_hat),axis=0),0)
        w_update = tf.matmul(tf.transpose(w_update_1),w_update_2)

        if l2_regularization:
            w_update = w_update + lamda * self.w

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (w_update)   ))
        update_w.append(tf.assign( self.v,self.v*beta2 + (1-beta2) * (w_update ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middle = learning_rate/(tf.sqrt(v_hat) + adam_e) * m_hat
        update_w.append(tf.assign(self.w,tf.add(self.w,adam_middle )))
        return w_update, update_w

# Func: k sparse auto encoders
class k_sparse_layer():

    def __init__(self,inc,outc,special_init=False):
        if special_init:
            interval = np.sqrt(6.0 / (inc + outc + 1.0))
            self.w  = tf.Variable(tf.random_uniform(shape=(inc, outc),minval=-interval,maxval=interval,dtype=tf.float64,seed=4))
        else:
            self.w = tf.Variable(tf.random_normal([inc,outc], stddev=0.05,seed=2,dtype=tf.float64))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,k_value = 1):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.topk_value = tf.nn.top_k(self.layer, k_value)
        self.topk_masks = tf.cast(tf.greater_equal(self.layer , tf.reduce_min(self.topk_value.values)),tf.float64)
        self.layerA = self.layer * self.topk_masks
        self.reconstructed_layer = tf.matmul(self.layerA,tf.transpose(self.w))
        return self.reconstructed_layer

    def backprop(self,gradient,l2_regularization=False):
        grad_part_1 = gradient
        grad_part_3 = self.layerA

        grad_middle = grad_part_1
        grad = tf.matmul(tf.transpose(grad_middle),grad_part_3)/batch_size

        if l2_regularization:
            grad = grad + lamda * self.w

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v,self.v*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middle = learning_rate/(tf.sqrt(v_hat) + adam_e) * m_hat
        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middle )))

        return grad,update_w

# Func: Layer for Sparse Filtering
class Sparse_Filter_Layer():

    def __init__(self,outc,changec):
        self.w = tf.Variable(tf.random_normal([outc,changec],stddev=1.0,seed=2,dtype=tf.float64))
        self.epsilon = 1e-20

    def getw(self): return self.w

    def soft_abs(self,value):
        return tf.sqrt(value ** 2 + self.epsilon)

    def feedforward(self,input):
        self.sparse_layer  = tf.matmul(input,self.w)
        second = self.soft_abs(self.sparse_layer )
        third  = tf.divide(second,tf.sqrt(tf.reduce_sum(second**2,axis=0)+self.epsilon))
        four = tf.divide(third,tf.sqrt(tf.reduce_sum(third**2,axis=1)[:,tf.newaxis] +self.epsilon))
        self.cost_update = tf.reduce_mean(four)
        return self.sparse_layer ,self.cost_update

# Func: Layer for self organizing maps
class SOM_Layer():

    def __init__(self,m,n,dim,num_epoch,learning_rate_som = 0.04,radius_factor = 1.1, gaussian_std=0.5):

        self.m = m
        self.n = n
        self.dim = dim
        self.gaussian_std = gaussian_std
        self.num_epoch = num_epoch
        # self.map = tf.Variable(tf.random_uniform(shape=[m*n,dim],minval=0,maxval=1,seed=2))
        self.map = tf.Variable(tf.random_normal(shape=[m*n,dim],seed=2))

        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
        self.alpha = learning_rate_som
        self.sigma = max(m,n)*1.1

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def getmap(self): return self.map
    def getlocation(self): return self.bmu_locs
    def feedforward(self,input):

        self.input = input
        self.grad_pass = tf.pow(tf.subtract(tf.expand_dims(self.map, axis=0),tf.expand_dims(self.input, axis=1)), 2)
        self.squared_distance = tf.reduce_sum(self.grad_pass, 2)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        self.bmu_locs = tf.reshape(tf.gather(self.location_vects, self.bmu_indices), [-1, 2])

    def backprop(self,iter,num_epoch):

        # Update the weigths
        radius = tf.subtract(self.sigma,
                                tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                    tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        alpha = tf.subtract(self.alpha,
                            tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                      tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        self.bmu_distance_squares = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2),
            2)

        self.neighbouaimed_sparsityod_func = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, "float32")), tf.multiply(
                tf.square(tf.multiply(radius, self.gaussian_std)), 2)))

        self.learning_rate_op = tf.multiply(self.neighbouaimed_sparsityod_func, alpha)

        self.numerator = tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.learning_rate_op, axis=-1),
            tf.expand_dims(self.input, axis=1)), axis=0)

        self.denominator = tf.expand_dims(
            tf.reduce_sum(self.learning_rate_op,axis=0) + float(1e-20), axis=-1)

        self.new_weights = tf.div(self.numerator, self.denominator)
        self.update = [tf.assign(self.map, self.new_weights)]

        return self.update,tf.reduce_mean(self.grad_pass, 1)

# Func: Layer for principal component analysis
class PCA_Layer():

    def __init__(self,dim,channel):

        self.alpha = tf.Variable(tf.random_normal(shape=[dim//2,dim//2,channel],dtype=tf.float32,stddev=0.05))
        self.beta  = tf.Variable(tf.ones(shape=[channel],dtype=tf.float32))

        self.current_sigma = None
        self.moving_sigma = tf.Variable(tf.zeros(shape=[(dim*dim*channel),(dim*dim*channel)//4],dtype=tf.float32))

    def feedforward(self,input,is_training):
        update_sigma = []

        # 1. Get the input Shape and reshape the tensor into [Batch,Dim]
        width,channel = input.shape[1],input.shape[3]
        reshape_input = tf.reshape(input,[batch_size,-1])
        trans_input = reshape_input.shape[1]

        # 2. Perform SVD and get the sigma value and get the sigma value
        singular_values, u, _ = tf.svd(reshape_input,full_matrices=False)

        def training_fn():
            # 3. Training
            sigma1 = tf.diag(singular_values)
            sigma = tf.slice(sigma1, [0,0], [trans_input, (width*width*channel)//4])
            pca = tf.matmul(u, sigma)
            update_sigma.append(tf.assign(self.moving_sigma,self.moving_sigma*0.9 + sigma* 0.1 ))
            return pca,update_sigma

        def testing_fn():
            # 4. Testing calculate hte pca using the Exponentially Weighted Moving Averages
            pca = tf.matmul(u, self.moving_sigma)
            return pca,update_sigma

        pca,update_sigma = tf.cond(is_training, true_fn=training_fn, false_fn=testing_fn)
        pca_reshaped = tf.reshape(pca,[batch_size,(width//2),(width//2),channel])
        out_put = self.alpha * pca_reshaped +self.beta

        return out_put,update_sigma

# Func: PCA Layer 2 no distingusing between training and testing
class tf_pca_layer():
    
    def __init__(self,n_components):
        self.n_components = tf.Variable(n_components)

    def feedforward(self,input):
        self.input = input
        self.cov = tf.matmul(self.input,tf.transpose(self.input)) / (input.shape[0].value-1)
        self.eigval,self.pc = tf.linalg.eigh(self.cov+10e-3)
        self.pc_projection = self.pc[:,-self.n_components:]
        self.layer = tf.matmul(tf.transpose(self.pc_projection),input)
        return self.layer
    def backprop(self,grad):
        mat_shape = self.input.shape[0].value
        diff = mat_shape - self.n_components
        d_pc_project = tf.transpose(tf.matmul(grad,tf.transpose(self.input)))
        added_mat = tf.zeros([mat_shape,diff],dtype=tf.float64)
        d_pc = tf.concat([d_pc_project,added_mat],1)
        E = tf.matmul(tf.ones([mat_shape,1],dtype=tf.float64),tf.transpose(self.eigval)[tf.newaxis,:]) - \
            tf.matmul(self.eigval[:,tf.newaxis],tf.ones([1,mat_shape],dtype=tf.float64))
        F = 1.0/(E + tf.eye(mat_shape,dtype=tf.float64)) - tf.eye(mat_shape,dtype=tf.float64)
        d_cov = tf.matmul(tf.linalg.inv(tf.transpose(self.pc)),
                tf.matmul(F * (tf.matmul(tf.transpose(self.pc),d_pc)),tf.transpose(self.pc)))
        d_x = tf.matmul(self.pc_projection,grad) + \
              (tf.matmul(d_cov,self.input) + tf.matmul(tf.transpose(d_cov),self.input))/(mat_shape-1)
        return d_x  
    
    def feedforward_recon(self,input):
        self.input = input
        self.cov = tf.matmul(self.input,tf.transpose(self.input)) / (input.shape[0].value-1)
        self.eigval,self.pc = tf.linalg.eigh(self.cov+10e-3)
        self.pc_projection = self.pc[:,-self.n_components:]
        self.layer = tf.matmul(self.pc_projection,tf.matmul(tf.transpose(self.pc_projection),input))
        return self.layer
    def backprop_recon(self,grad):
        mat_shape = self.input.shape[0].value
        d_pc_project = tf.matmul(grad,tf.transpose(tf.matmul(tf.transpose(self.pc_projection),self.input)))  +\
                       tf.transpose(tf.matmul(tf.transpose(self.pc_projection),tf.matmul(grad,tf.transpose(self.input))))
        diff = mat_shape - self.n_components
        
        added_mat = tf.zeros([mat_shape,diff],dtype=tf.float64)
        d_pc = tf.concat([d_pc_project,added_mat],1)
        E = tf.matmul(tf.ones([mat_shape,1],dtype=tf.float64),tf.transpose(self.eigval)[tf.newaxis,:]) - \
            tf.matmul(self.eigval[:,tf.newaxis],tf.ones([1,mat_shape],dtype=tf.float64))
        F = 1.0/(E + tf.eye(mat_shape,dtype=tf.float64)) - tf.eye(mat_shape,dtype=tf.float64)
        d_cov = tf.matmul(tf.linalg.inv(tf.transpose(self.pc)),
                tf.matmul(F * (tf.matmul(tf.transpose(self.pc),d_pc)),tf.transpose(self.pc)))
        d_x = tf.matmul(tf.transpose(tf.matmul(self.pc_projection,tf.transpose(self.pc_projection))),grad) + \
              (tf.matmul(d_cov,self.input) + tf.matmul(tf.transpose(d_cov),self.input))/(mat_shape-1)
        return d_x  

# Func: KPCA layer using eigh
class KPCA_layer():

    def __init__(self):
        pass
    def getw(self): return self.small_eigvec
    def feedforward(self,input,n_components,gamma = 15):
        self.input = input
        self.distance_matrix = -2 * tf.matmul(input,tf.transpose(input)) + tf.reduce_sum(input**2,axis=1) + tf.reduce_sum(input**2,axis=1)[:,tf.newaxis]
        self.k = tf.exp(-gamma * self.distance_matrix)
        N = self.k.shape[0].value
        ones = tf.ones([N,N],dtype=tf.float64) / N
        self.center_k = self.k - tf.matmul(ones,self.k) - tf.matmul(self.k,ones) + tf.matmul(tf.matmul(ones,self.k),ones)
        self.eigval,self.eigvec = tf.linalg.eigh(self.center_k)
        self.small_eigvec = self.eigvec[:,-n_components:]
        return self.small_eigvec

    def backprop_upsuper(self):
        pass

    def backprop(self,grad):
        pass

class FastICA_Layer():
    """Performs ICA via FastICA method

    Parameters
    ----------
    inc : type
        Description of parameter `inc`.
    outc : type
        Description of parameter `outc`.
    act : type
        Description of parameter `act`.
    d_act : type
        Description of parameter `d_act`.

    Attributes
    ----------
    w : type
        Description of attribute `w`.
    sym_decorrelation : type
        Description of attribute `sym_decorrelation`.
    m : type
        Description of attribute `m`.
    v : type
        Description of attribute `v`.
    self,matrix : type
        Description of attribute `self,matrix`.
    act
    d_act

    """

    def __init__(self,inc,outc,act,d_act):
        self.w = tf.Variable(self.sym_decorrelation(tf.random_normal(shape=[inc,outc],stddev=0.05,dtype=tf.float64,seed=2)))
        self.m = tf.Variable(tf.zeros_like(self.w)) ; self.v = tf.Variable(tf.zeros_like(self.w))
        self.act = act; self.d_act = d_act

    def sym_decorrelation(self,matrix):
        s, u = tf.linalg.eigh(tf.matmul(matrix,tf.transpose(matrix)))
        decor_matrx = tf.matmul(u * (1.0/tf.sqrt(s)),tf.transpose(u))
        return tf.matmul(decor_matrx,matrix)

    def getw(self): return self.w

    def feedforward(self,input):
        self.input = input
        self.layer = tf.matmul(self.w,input)
        return self.layer

    def backprop_ica(self):
        self.layerA  = self.act(tf.matmul(self.w,self.input))
        self.layerDA = tf.reduce_mean(self.d_act(tf.matmul(self.w,self.input)),-1)
        grad_pass = tf.matmul(tf.transpose(self.w),self.layer)

        grad_w = tf.matmul(self.layerA,tf.transpose(self.input)) / self.input.shape[1].value - self.layerDA[:,tf.newaxis] * self.w
        grad = self.sym_decorrelation(grad_w)

        update_w = []

        # ==== Correct Method of Weight Update ====
        # update_w.append(tf.assign(self.w,grad))

        # ==== Wrong (gradient ascent) Method of Weight Update ====
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v,self.v*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middle = m_hat * 0.01 *  learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middle )))
        return grad_pass,update_w

class ICA_Layer():
    """Perform Independent component analysis via gradient ascent method
    see: http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_ICA09.pdf

    Parameters
    ----------
    inc : type
        Description of parameter `inc`.
    act : type
        Description of parameter `act`.
    d_act : type
        Description of parameter `d_act`.

    Attributes
    ----------
    w : type
        Description of attribute `w`.
    m : type
        Description of attribute `m`.
    v : type
        Description of attribute `v`.
    self,input : type
        Description of attribute `self,input`.
    act
    d_act

    """

    def __init__(self,inc,act,d_act):
        self.w = tf.Variable(tf.random_normal([inc,inc],stddev=0.05,seed=2,dtype=tf.float64))
        self.m = tf.Variable(tf.zeros_like(self.w));self.v = tf.Variable(tf.zeros_like(self.w));
        self.act = act; self.d_act = d_act

    def feedforward(self,input):
        self.input       = input
        self.ica_est     = tf.matmul(input,self.w)
        self.ica_est_act = self.act(self.ica_est)
        return self.ica_est_act,self.w

    def backprop(self):
        grad_part_2 = self.d_act(self.ica_est)
        grad_part_3 = self.input

        grad_pass = tf.matmul(grad_part_2,tf.transpose(self.w))
        grad_sum_1 = tf.expand_dims(tf.reduce_sum(tf.transpose(self.input),1),1) / batch_size
        grad_sum_2 = tf.expand_dims(tf.reduce_sum(self.ica_est_act,0),0) / batch_size
        grad = tf.linalg.inv(tf.transpose(self.w)) - (2.0/batch_size) * tf.matmul(grad_sum_1,grad_sum_2)

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v,self.v*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middle = m_hat *  learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,self.w+adam_middle))
        return grad_pass,update_w

class tf_mean_layer():

    def __init__(self):          
        pass

    def feedforward(self,input):
        self.mean = tf.reduce_mean(input,1)
        return input-self.mean[:,tf.newaxis]

    def backprop(self,grad):
        return grad * (1 - 1.0/grad.shape[0].value)

class zca_whiten_layer():
    """ZCA Whiten operation layer.

    Parameters
    ----------
    self) : pass    def feedforward(self
        Description of parameter `self)`.
    input : type
        Description of parameter `input`.
    EPS : type
        Description of parameter `EPS`.

    Attributes
    ----------
    sigma : type
        Description of attribute `sigma`.
    eigenval : type
        Description of attribute `eigenval`.
    eigvector : type
        Description of attribute `eigvector`.
    U : type
        Description of attribute `U`.
    whiten : type
        Description of attribute `whiten`.
    self,grad : type
        Description of attribute `self,grad`.
    input

    """

    def __init__(self): pass

    def feedforward(self,input,EPS=10e-5):
        self.input = input
        self.sigma = tf.matmul(tf.transpose(input),input) / input.shape[0].value
        self.eigenval,self.eigvector = tf.linalg.eigh(self.sigma)
        self.U = tf.matmul(tf.matmul(self.eigvector,tf.diag(1./ tf.sqrt(self.eigenval+EPS))),tf.transpose(self.eigvector))
        self.whiten = tf.matmul(input,self.U)
        return self.whiten

    def backprop(self,grad,EPS=10e-5):
        d_U = tf.matmul(tf.transpose(self.input),grad)

        # d_eig_value = self.eigvector.T.dot(d_U).dot(self.eigvector) * (-0.5) * np.diag(1. / (self.eigenval+EPS) ** 1.5)
        d_eig_value = tf.matmul(tf.matmul(tf.transpose(self.eigvector),d_U),self.eigvector) * (-0.5) * tf.diag(1./(self.eigvector+EPS) ** 1.5 )

        # d_eig_vector = d_U.dot( (np.diag(1. / np.sqrt(self.eigenval+EPS)).dot(self.eigvector.T)).T  ) + (self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)))).dot(d_U)
        d_eig_vector = tf.matmul(d_U,tf.transpose(tf.matmul( tf.diag(1./ tf.sqrt(self.eigenval+EPS)), tf.transpose(self.eigvector)))) + \
                       tf.matmul(tf.transpose(d_U),tf.matmul(self.eigvector,tf.diag(1./ tf.sqrt(self.eigenval+EPS))))

        # E = np.ones((grad.shape[1],1)).dot(np.expand_dims(self.eigenval.T,0)) - np.expand_dims(self.eigenval,1).dot(np.ones((1,grad.shape[1])))
        E = tf.matmul(tf.ones([grad.shape[0].value,1],dtype=tf.float64),tf.transpose(self.eigenval)[tf.newaxis,:]) - \
            tf.matmul(self.eigenval[:,tf.newaxis],tf.ones([1,grad.shape[0].value],dtype=tf.float64))

        # K_matrix = 1./(E + np.eye(grad.shape[1])) - np.eye(grad.shape[1])
        K_matrix = 1.0 /( E + tf.eye(grad.shape[1].value,dtype=tf.float64)) - tf.eye(grad.shape[1].value,dtype=tf.float64)

        # np.fill_diagonal(d_eig_value,0.0)
        tf.matrix_set_diag(d_eig_value,0.0)

        # d_sigma = self.eigvector.dot(
        #             K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + d_eig_value
        #             ).dot(self.eigvector.T)
        d_sigma = tf.matmul(tf.matmul(self.eigvector,
                    tf.transpose(K_matrix) * tf.matmul(tf.transpose(self.eigvector),d_eig_vector) + d_eig_value),
                    tf.transpose(self.eigvector))

        # d_x = grad.dot(self.U.T) + (2./grad.shape[0]) * self.input.dot(d_sigma) * 2
        d_x  = tf.matmul(grad,tf.transpose(self.U)) + (2.0/grad.shape[0].value) * tf.matmul(self.input,d_sigma) * 2

        return d_x

# Stacked Denoising Auto encoder layers
class FNN_Stacked_Denoising():
    # From: Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion

    def __init__(self,inc,outc,act=tf_sigmoid,d_act=d_tf_sigmoid,special_init=True,which_reg=0.0):
        if special_init:
            interval = 1/np.sqrt(inc)
            self.w   = tf.Variable(tf.random_uniform(shape=(inc, outc),minval=-interval,maxval=interval,dtype=tf.float32,seed=2))
        else:
            self.w = tf.Variable(tf.random_normal([inc,outc], stddev=0.05,seed=2,dtype=tf.float32))

        self.m,self.v       = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act
        self.which_reg      = which_reg

    def getw(self): return self.w

    def feedforward(self,input=None):
        self.input  = input
        self.layer  = tf.matmul(input,self.w) 
        self.layerA = self.act(self.layer)
        return self.layerA
    
    def feedforward_inverse(self,input):
        self.input_inv = input
        self.layer_inv = tf.matmul(input,tf.transpose(self.w))
        self.layerA_inv= self.act(self.layer_inv)
        return self.layerA_inv

    def backprop(self,gradient=None,which_reg=0):
        
        grad_part1_inv = gradient
        grad_part2_inv = self.d_act(self.layer_inv)
        grad_part3_inv = self.input_inv
        grad_pass_inv  = grad_part1_inv * grad_part2_inv
        grad_inv       = tf.transpose(tf.transpose(grad_part3_inv) @ grad_pass_inv)
        
        grad_part1     = grad_pass_inv @ self.w
        grad_part2     = self.d_act(self.layer)
        grad_part3     = self.input
        grad_          = tf.transpose(grad_part3) @ (grad_part1 * grad_part2)
        
        grad           = (grad_inv + grad_)/batch_size

        # === Reg ===
        if self.which_reg == 0.5:    grad = grad + lamda * (tf.sqrt(tf.abs(self.w))) * (1.0/tf.sqrt(tf.abs(self.w)+ 10e-5)) * tf.sign(self.w)
        if self.which_reg == 1:      grad = grad + lamda * tf.sign(self.w)
        if self.which_reg == 1.5:    grad = grad + lamda * 1.0/(tf.sqrt(tf.square(self.w) + 10e-5)) * self.w
        if self.which_reg == 2:      grad = grad + lamda * (1.0/tf.sqrt(tf.square(tf.abs(self.w))+ 10e-5)) * tf.abs(self.w) * tf.sign(self.w)
        if self.which_reg == 2.5:    grad = grad + lamda * 2.0 * self.w
        if self.which_reg == 3:      grad = grad + lamda * tf.pow(tf.pow(tf.abs(self.w),3)+ 10e-5,-0.66) * tf.pow(tf.abs(self.w),2) * tf.sign(self.w)
        if self.which_reg == 4:      grad = grad + lamda * tf.pow(tf.pow(tf.abs(self.w),4)+ 10e-5,-0.75) * tf.pow(tf.abs(self.w),3) * tf.sign(self.w)

        # Update the Weight First
        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v,self.v*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middle = m_hat *  learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middle )))

        return update_w

class global_contrast_layer():
    
    def __init__(self,vector_shape,aimed_std=1.0):
        self.moving_mean = tf.Variable(tf.zeros(shape=(vector_shape,1),dtype=tf.float64))
        self.aimed_std   = tf.Variable(aimed_std,dtype=tf.float64)
        self.reg_lamda   = 10.0; self.EPS         = 1e-8
        
    def feedforward(self,input,training_phase):
        self.input      = input
        self.image_mean = tf.reduce_mean(self.input,-1)[:,None]
        self.centered   = self.input - self.image_mean
        self.square     = tf.square(self.centered)
        self.whole_mean = tf.reduce_mean(self.square,-1)[:,None]
        self.sqrt       = tf.sqrt(self.reg_lamda + self.whole_mean) + self.EPS
        
        def training_fn():
            gcn_data = self.aimed_std * (self.centered/self.sqrt)
            return gcn_data,tf.assign(self.moving_mean,self.moving_mean * 0.9 + 0.1 * self.image_mean)
        
        def  testing_fn():
            centered_data  = self.input - self.moving_mean
            squared_data   = tf.square(centered_data)
            whole_mean_data= tf.reduce_mean(squared_data,-1)[:,None]
            sqrt           = tf.sqrt(self.reg_lamda + whole_mean_data) + self.EPS
            gcn_data       = self.aimed_std * (centered_data/self.sqrt)
            return gcn_data, tf.assign(self.moving_mean,self.moving_mean)
        
        self.output,update_gcn_mean = tf.cond(training_phase,true_fn=training_fn,false_fn=testing_fn)
        return self.output,tf.assign(self.moving_mean,self.moving_mean)        
        
    def backprop(self,grad):
        grad_1 = grad
        grad_2 = self.aimed_std / self.sqrt
        grad_3 = -(self.centered ** 2 /self.sqrt ** 2) * (1.0/(self.sqrt-self.EPS)) / (self.input.shape[1].value)
        grad_pass = grad_1 * (grad_2 + grad_3) * (1-1./self.input.shape[1].value)
        return grad_pass


class Reconstructive_PCA():
    
    def __init__(self,n_components):
        self.n_components= tf.Variable(n_components,dtype=tf.int32)
        
    def feedforward(self,input):
        self.input  = input
        self.cov    = tf.matmul(self.input,tf.transpose(self.input)) / (self.input.shape[1].value-1)
        self.eigenvalues,self.eigenvectors  = tf.linalg.eigh(self.cov)
        self.projection_vector              = self.eigenvectors[:,-self.n_components:]
        self.reduced= tf.matmul(tf.transpose(self.projection_vector),self.input)
        self.reconst= tf.matmul(self.projection_vector,self.reduced)
        return self.reconst     
    
    def backprop(self,grad):
        grad1   = grad
        grad_A    = grad1 @ tf.transpose(tf.transpose(self.projection_vector) @ self.input) + \
                     tf.transpose(tf.transpose(self.projection_vector)@grad1@tf.transpose(self.input))
        diff      = self.input.shape[0].value - self.n_components
        added_mat = tf.zeros([self.input.shape[0].value,diff],dtype=tf.float64)
        grad_A_add= tf.concat([added_mat,grad_A],1)
        
        dia_eig   = tf.ones((self.input.shape[0].value,1),dtype=tf.float64) @ tf.transpose(self.eigenvalues)[None,:] - \
                    self.eigenvalues[:,None] @ tf.ones((1,self.input.shape[0].value),dtype=tf.float64)
        diag_k    = 1./(dia_eig + tf.eye(self.input.shape[0].value,dtype=tf.float64)) - tf.eye(self.input.shape[0].value,dtype=tf.float64)
        
        grad_B    = self.eigenvectors @ ( tf.transpose(diag_k)* (tf.transpose(self.eigenvectors)@ grad_A_add)) @ tf.transpose(self.eigenvectors)
        grad_B    = 0.5 * (tf.transpose(grad_B) + grad_B)
        grad_c    = (grad_B @ self.input + tf.transpose(tf.transpose(self.input) @ grad_B))/(self.input.shape[1].value-1)
        return grad_c

class tf_min_max_layer():
    
    def __init__(self,vector_shape,user_max=1.0,user_min=0.0):
        self.moving_min = tf.Variable(tf.zeros(shape=(vector_shape,1),dtype=tf.float64))
        self.moving_max = tf.Variable(tf.zeros(shape=(vector_shape,1),dtype=tf.float64))
        self.user_min   = tf.Variable(user_min,dtype=tf.float64); 
        self.user_max   = tf.Variable(user_max,dtype=tf.float64); 
        
    def feedforward(self,input,training_phase):
        self.input    = input
        self.min_vec  = tf.reduce_min(input,-1)[:,None]
        self.min_index= tf.argmin(input,-1)
        self.max_vec  = tf.reduce_max(input,-1)[:,None]
        self.max_index= tf.argmax(input,-1)
        
        def training_fn():
            normalized_data = (self.user_max-self.user_min)  * \
            ((self.input - self.min_vec)/(self.max_vec - self.min_vec))          + self.user_min
            
            update_min_max = []
            update_min_max.append(tf.assign(self.moving_min,self.moving_min * 0.9 + 0.1 * self.min_vec))
            update_min_max.append(tf.assign(self.moving_max,self.moving_max * 0.9 + 0.1 * self.max_vec))
            return normalized_data,update_min_max
        
        # Testing Moving Average Mean        
        def  testing_fn():
            normalized_data = (self.user_max-self.user_min) * \
            ((self.input - self.moving_min)/(self.moving_max - self.moving_min)) + self.user_min
            
            update_min_max = []
            update_min_max.append(tf.assign(self.moving_min,self.moving_min))
            update_min_max.append(tf.assign(self.moving_max,self.moving_max))
            return normalized_data,update_min_max
        
        self.output,update_min_max = tf.cond(training_phase,true_fn=training_fn,false_fn=testing_fn)
        return self.output,update_min_max
    
    def backprop(self,grad):
        grad1   = grad
        
        # Create Mask for min / max value for row
        indices = tf.range(0, self.input.shape[0].value,dtype=tf.int64)
        min_indices = tf.stack([indices, self.min_index], axis=1)
        max_indices = tf.stack([indices, self.max_index], axis=1)
        grad_min = tf.cast(tf.sparse_to_dense(min_indices, self.input.shape, sparse_values=1, default_value=0),dtype=tf.float64)
        grad_max = tf.cast(tf.sparse_to_dense(max_indices, self.input.shape, sparse_values=1, default_value=0),dtype=tf.float64)
        
        grad_max_min = 1.0/(self.max_vec-self.min_vec)
        grad_pass    = grad1 * (self.user_max-self.user_min) * (
            grad_max_min + \
            (self.input - self.max_vec)/tf.square(grad_max_min) * grad_min + \
            (self.min_vec - self.input)/tf.square(grad_max_min) * grad_max
        )
        
        # Again do not RETURN grad_magrad_passx_min correct gradient is grad_pass
        return grad_pass        

class tf_batch_norm():
    
    def __init__(self,height,width,channel_size):
        self.height = height; self.width = width
        self.moving_mean = tf.Variable(tf.zeros((1,1,1,channel_size),dtype=tf.float64))
        self.moving_var  = tf.Variable(tf.zeros((1,1,1,channel_size),dtype=tf.float64))
        
    def feedforward(self,input,training_phase,eplison = 10e-8):
        self.input = input 
        self.mean  = tf.reduce_mean(self.input,(0,1,2),keep_dims=True) 
        self.center= self.input - self.mean
        self.var   = tf.reduce_mean(tf.square(self.center),(0,1,2),keep_dims=True)
        self.stand = self.center/tf.sqrt(self.var + eplison)
        
        def training_fn():
            stand         = self.stand
            update_moving = []
            update_moving.append(tf.assign(self.moving_mean,self.moving_mean * 0.9 + 0.1 * self.mean))
            update_moving.append(tf.assign(self.moving_var, self.moving_var * 0.9  + 0.1 * self.var))
            return stand,update_moving
        
        def testing_fn():
            centered_data  = self.input - self.moving_mean
            stand          = centered_data/tf.sqrt(self.moving_var + eplison)
            update_moving = []
            update_moving.append(tf.assign(self.moving_mean,self.moving_mean))
            update_moving.append(tf.assign(self.moving_var, self.moving_var))
            return stand, update_moving
        
        self.output,update_batch = tf.cond(training_phase,true_fn=training_fn,false_fn=testing_fn)
        return self.output,update_batch 
    
    def backprop(self,grad,eplison = 10e-8):
        grad_c = grad * self.center * (-1) * (tf.pow(self.var+eplison,-1.5))
        grad_b = tf.reduce_mean(self.center,(0,1,2),keep_dims=True)
        grad_bb= grad * 1.0/tf.sqrt(self.var + eplison)
        grad   = (grad_c * grad_b + grad_bb) * (1.0-1.0/(batch_size * self.height ** 2)) 
        return grad



class tf_instance_norm_layer():
    
    def __init__(self,batch_size,vector_shape,axis):
        self.moving_mean = tf.Variable(tf.zeros(shape=[batch_size,1,1,vector_shape],dtype=tf.float64))
        self.moving_vari = tf.Variable(tf.zeros(shape=[batch_size,1,1,vector_shape],dtype=tf.float64))
        self.axis        = axis
        
    def feedforward(self,input,training_phase=True,eps = 1e-8):
        self.input = input
        self.input_size          = self.input.shape
        self.batch,self.h,self.w,self.c = self.input_size[0].value,self.input_size[1].value,self.input_size[2].value,self.input_size[3].value

        # Training Moving Average Mean         
        def training_fn():
            self.mean    = tf.reduce_mean(self.input,axis=self.axis ,keepdims=True)
            self.var     = tf.reduce_mean(tf.square(self.input-self.mean),axis=self.axis,keepdims=True)
            centered_data= (self.input - self.mean)/tf.sqrt(self.var + eps)
            
            update_variable = []
            update_variable.append(tf.assign(self.moving_mean,self.moving_mean*0.9 + 0.1 * self.mean ))
            update_variable.append(tf.assign(self.moving_vari,self.moving_vari*0.9 + 0.1 * self.var  ))
            return centered_data,update_variable
        
        # Testing Moving Average Mean        
        def  testing_fn():
            centered_data   = (self.input - self.moving_mean)/tf.sqrt(self.moving_vari + eps)
            update_variable = []
            update_variable.append(tf.assign(self.moving_mean,self.moving_mean))
            update_variable.append(tf.assign(self.moving_vari,self.moving_vari))
            return centered_data,update_variable
        
        self.output,update_variable = tf.cond(training_phase,true_fn=training_fn,false_fn=testing_fn)
        return self.output,update_variable
    
    def backprop(self,grad,eps = 1e-8):
        change_parts = 1.0 /(self.h * self.w)
        grad_sigma   = tf.reduce_sum( grad *  (self.input-self.mean)     ,axis=self.axis,keepdims=True) * -0.5 * (self.var+eps) ** -1.5
        grad_mean    = tf.reduce_sum( grad *  (-1./tf.sqrt(self.var+eps)),axis=self.axis,keepdims=True) + grad_sigma * change_parts * 2.0 * tf.reduce_sum((self.input-self.mean),axis=self.axis,keepdims=True) * -1
        grad_x       = grad * 1/(tf.sqrt(self.var+eps)) + grad_sigma * change_parts * 2.0 * (self.input-self.mean) + grad_mean * change_parts
        return grad_x



class tf_layer_norm_layer():
    
    def __init__(self,vector_shape,axis):
        self.moving_mean = tf.Variable(tf.zeros(shape=[vector_shape,1,1,1],dtype=tf.float64))
        self.moving_vari = tf.Variable(tf.zeros(shape=[vector_shape,1,1,1],dtype=tf.float64))
        self.axis        = axis
        
    def feedforward(self,input,training_phase=True,eps = 1e-8):
        self.input = input
        self.input_size          = self.input.shape
        self.batch,self.h,self.w,self.c = self.input_size[0].value,self.input_size[1].value,self.input_size[2].value,self.input_size[3].value

        # Training Moving Average Mean         
        def training_fn():
            self.mean    = tf.reduce_mean(self.input,axis=self.axis ,keepdims=True)
            self.var     = tf.reduce_mean(tf.square(self.input-self.mean),axis=self.axis,keepdims=True)
            centered_data= (self.input - self.mean)/tf.sqrt(self.var + eps)
            
            update_variable = []
            update_variable.append(tf.assign(self.moving_mean,self.moving_mean*0.9 + 0.1 * self.mean ))
            update_variable.append(tf.assign(self.moving_vari,self.moving_vari*0.9 + 0.1 * self.var  ))
            return centered_data,update_variable
        
        # Testing Moving Average Mean        
        def  testing_fn():
            centered_data   = (self.input - self.moving_mean)/tf.sqrt(self.moving_vari + eps)
            update_variable = []
            update_variable.append(tf.assign(self.moving_mean,self.moving_mean))
            update_variable.append(tf.assign(self.moving_vari,self.moving_vari))
            return centered_data,update_variable
        
        self.output,update_variable = tf.cond(training_phase,true_fn=training_fn,false_fn=testing_fn)
        return self.output,update_variable
    
    def backprop(self,grad,eps = 1e-8):
        change_parts = 1.0 /(self.h * self.w * self.c)
        grad_sigma   = tf.reduce_sum( grad *  (self.input-self.mean)     ,axis=self.axis,keepdims=True) * -0.5 * (self.var+eps) ** -1.5
        grad_mean    = tf.reduce_sum( grad *  (-1./tf.sqrt(self.var+eps)),axis=self.axis,keepdims=True) + grad_sigma * change_parts * 2.0 * tf.reduce_sum((self.input-self.mean),axis=self.axis,keepdims=True) * -1
        grad_x       = grad * 1/(tf.sqrt(self.var+eps)) + grad_sigma * change_parts * 2.0 * (self.input-self.mean) + grad_mean * change_parts
        return grad_x

class tf_batch_norm_layer():
    
    def __init__(self,vector_shape,axis):
        self.moving_mean = tf.Variable(tf.zeros(shape=[1,1,1,vector_shape],dtype=tf.float64))
        self.moving_vari = tf.Variable(tf.zeros(shape=[1,1,1,vector_shape],dtype=tf.float64))
        self.axis        = axis
        
    def feedforward(self,input,training_phase=True,eps = 1e-8):
        self.input = input
        self.input_size          = self.input.shape
        self.batch,self.h,self.w,self.c = self.input_size[0].value,self.input_size[1].value,self.input_size[2].value,self.input_size[3].value

        # Training Moving Average Mean         
        def training_fn():
            self.mean    = tf.reduce_mean(self.input,axis=self.axis ,keepdims=True)
            self.var     = tf.reduce_mean(tf.square(self.input-self.mean),axis=self.axis,keepdims=True)
            centered_data= (self.input - self.mean)/tf.sqrt(self.var + eps)
            
            update_variable = []
            update_variable.append(tf.assign(self.moving_mean,self.moving_mean*0.9 + 0.1 * self.mean ))
            update_variable.append(tf.assign(self.moving_vari,self.moving_vari*0.9 + 0.1 * self.var  ))
            return centered_data,update_variable
        
        # Testing Moving Average Mean        
        def  testing_fn():
            centered_data   = (self.input - self.moving_mean)/tf.sqrt(self.moving_vari + eps)
            update_variable = []
            update_variable.append(tf.assign(self.moving_mean,self.moving_mean))
            update_variable.append(tf.assign(self.moving_vari,self.moving_vari))
            return centered_data,update_variable
        
        self.output,update_variable = tf.cond(training_phase,true_fn=training_fn,false_fn=testing_fn)
        return self.output,update_variable
    
    def backprop(self,grad,eps = 1e-8):
        change_parts = 1.0 /(self.batch * self.h * self.w)
        grad_sigma   = tf.reduce_sum( grad *  (self.input-self.mean)     ,axis=self.axis,keepdims=True) * -0.5 * (self.var+eps) ** -1.5
        grad_mean    = tf.reduce_sum( grad *  (-1./tf.sqrt(self.var+eps)),axis=self.axis,keepdims=True) + grad_sigma * change_parts * 2.0 * tf.reduce_sum((self.input-self.mean),axis=self.axis,keepdims=True) * -1
        grad_x       = grad * 1/(tf.sqrt(self.var+eps)) + grad_sigma * change_parts * 2.0 * (self.input-self.mean) + grad_mean * change_parts
        return grad_x
# ================= LAYER CLASSES =================
