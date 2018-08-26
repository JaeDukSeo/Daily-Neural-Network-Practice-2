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
from scipy.signal import convolve

np.random.seed(0)
ia.random.seed(0)
np.set_printoptions(precision = 3,suppress =True)
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

# def: relu activations
def np_relu(x): return x * (x > 0)
def d_np_relu(x): return 1. * (x > 0)
def np_tanh(x): return  np.tanh(x)
def d_np_tanh(x): return 1. - np_sigmoid(x) ** 2
def np_sigmoid(x): return  1/(1+np.exp(-x))
def d_np_sigmoid(x): return np_sigmoid(x) * (1.-np_sigmoid(x))

# def: fully connected layer
r = np.random.RandomState(1234)
class np_FNN():

    def __init__(self,inc,outc,batch_size,act=np_relu,d_act = d_np_relu):
        self.w = r.normal(0,0.01,size=(inc, outc))
        self.b = r.normal(0,0.005,size=(outc))
        self.m,self.v = np.zeros_like(self.w),np.zeros_like(self.w)
        self.mb,self.vb = np.zeros_like(self.b),np.zeros_like(self.b)
        self.act = act; self.d_act = d_act
        self.batch_size = batch_size

    def getw(self): return self.w

    def feedforward(self,input):
        self.input  = input
        self.layer  = self.input.dot(self.w) + self.b
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,grad,lr_rate,reg=True):
        grad_1 = grad
        grad_2 = self.d_act(self.layer)
        grad_3 = self.input

        grad_middle = grad_1 * grad_2
        grad_b = grad_middle.sum(0) / grad.shape[0]
        grad = grad_3.T.dot(grad_middle) / grad.shape[0]
        grad_pass = grad_middle.dot(self.w.T)

        if reg:
            grad = grad + lamda * self.w
            grad_b = grad_b + lamda * self.b

        self.m = self.m * beta1 + (1. - beta1) * grad
        self.v = self.v * beta2 + (1. - beta2) * grad ** 2
        m_hat,v_hat = self.m/(1.-beta1), self.v/(1.-beta2)
        adam_middle =  m_hat * lr_rate / (np.sqrt(v_hat) + adam_e)
        self.w = self.w - adam_middle

        self.mb = self.mb * beta1 + (1. - beta1) * grad_b
        self.vb = self.vb * beta2 + (1. - beta2) * grad_b ** 2
        m_hatb,v_hatb = self.mb/(1.-beta1), self.vb/(1.-beta2)
        adam_middleb =  m_hatb * lr_rate /(np.sqrt(v_hatb) + adam_e)
        self.b = self.b - adam_middleb

        return grad_pass

# def: centering layer
class centering_layer():

    def __init__(self): pass

    def feedforward(self,input):
        x_mean = np.sum(input,axis=0) / input.shape[0]
        return input - x_mean

    def backprop(self,grad):
        return grad * ( 1. - 1./grad.shape[0])

# def: Batch Normalization
class standardization_layer():

    def __init__(self): pass

    def feedforward(self,input,EPS=10e-5):
        self.input = input
        self.mean  = np.sum(input,axis=0) / input.shape[0]
        self.std   = np.sum( (self.input - self.mean) ** 2,0)  / input.shape[0]
        self.x_hat = (input - self.mean) / np.sqrt(self.std + EPS)
        return self.x_hat

    def backprop(self,grad,EPS=10e-5):
        dem = 1./(grad.shape[0] * np.sqrt(self.std + EPS ) )
        d_x = grad.shape[0] * grad - np.sum(grad,axis = 0) - self.x_hat*np.sum(grad*self.x_hat, axis=0)
        return d_x * dem

# def: zca whitening layer
class zca_whiten_layer():

    def __init__(self): pass

    def feedforward(self,input,EPS=1e-10):
        self.input = input
        self.sigma = input.T.dot(input) / input.shape[0]
        self.eigenval,self.eigvector = np.linalg.eigh(self.sigma)
        self.U = self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS))).dot(self.eigvector.T)
        self.whiten = input.dot(self.U)
        return self.whiten

    def backprop(self,grad,EPS=1e-10):
        d_U = self.input.T.dot(grad)
        d_eig_value = self.eigvector.T.dot(d_U).dot(self.eigvector) * (-0.5) * np.diag(1. / (self.eigenval+EPS) ** 1.5)
        d_eig_vector = d_U.dot( (np.diag(1. / np.sqrt(self.eigenval+EPS)).dot(self.eigvector.T)).T  ) + (self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)))).dot(d_U)
        E = np.ones((grad.shape[1],1)).dot(np.expand_dims(self.eigenval.T,0)) - np.expand_dims(self.eigenval,1).dot(np.ones((1,grad.shape[1])))
        K_matrix = 1./(E + np.eye(grad.shape[1])) - np.eye(grad.shape[1])
        np.fill_diagonal(d_eig_value,0.0)
        d_sigma = self.eigvector.dot(
                    K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + d_eig_value
                    ).dot(self.eigvector.T)
        d_x = grad.dot(self.U.T) + (2./grad.shape[0]) * self.input.dot(d_sigma) * 2
        return d_x

# def: soft max function for 2D
def stable_softmax(x,axis=None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=1)[:,np.newaxis])
    return e_x / e_x.sum(axis=1)[:,np.newaxis]

def conv_forward(X, W, b, stride=1, padding=1):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)
    cache = (X, W, b, stride, padding, X_col)
    return out,cache

def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dW, db
# mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
mnist = input_data.read_data_sets('../../Dataset/fashionmnist/',one_hot=True)
train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
train_data =  np.vstack((train_data,mnist.validation.images))
train_label = np.vstack((train_label,mnist.validation.labels))

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

train_data = train_data.reshape(60000,28,28)[:,np.newaxis,:,:]
test_data  =  test_data.reshape(10000,28,28)[:,np.newaxis,:,:]
def maxpool_forward(X, size=2, stride=2):
    def maxpool(X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    return _pool_forward(X, maxpool, size, stride)

def maxpool_backward(dout, cache):
    def dmaxpool(dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col

    return _pool_backward(dout, dmaxpool, cache)

def _pool_forward(X, pool_fun, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    out, pool_cache = pool_fun(X_col)

    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)

    cache = (X, size, stride, X_col, pool_cache)

    return out, cache

def _pool_backward(dout, dpool_fun, cache):
    X, size, stride, X_col, pool_cache = cache
    n, d, w, h = X.shape

    dX_col = np.zeros_like(X_col)
    dout_col = dout.transpose(2, 3, 0, 1).ravel()

    dX = dpool_fun(dX_col, dout_col, pool_cache)

    dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
    dX = dX.reshape(X.shape)

    return dX

# hyper
num_epoch = 30
batch_size = 500
learning_rate = 0.0005
lamda = 0.0008
print_size  = 1
beta1,beta2,adam_e = 0.9,0.999,1e-20

filter_size_0 = 3
filter_size_1 = 3
layer0_output_c = 8
layer1_output_c = 16

# class of layers
l0_w = r.normal(0,0.01, size=(layer0_output_c,1,filter_size_0,filter_size_0));               l0_b = r.normal(0,0.005,size=(layer0_output_c,1))
l0_w_m,l0_w_v = np.zeros_like(l0_w),np.zeros_like(l0_w)
l0_b_m,l0_b_v = np.zeros_like(l0_b),np.zeros_like(l0_b)
l0_special = zca_whiten_layer()

l1_w = r.normal(0,0.01, size=(layer1_output_c,layer0_output_c,filter_size_1,filter_size_1)); l1_b = r.normal(0,0.005,size=(layer1_output_c,1))
l1_w_m,l1_w_v = np.zeros_like(l1_w),np.zeros_like(l1_w)
l1_b_m,l1_b_v = np.zeros_like(l1_b),np.zeros_like(l1_b)
l2 = np_FNN(layer1_output_c*7*7,10    ,batch_size,act=np_relu,d_act=d_np_relu)

# train
for iter in range(num_epoch):

    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]

    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    train_data,train_label = shuffle(train_data,train_label)

    for current_data_index in range(0,len(train_data),batch_size):
        current_data = train_data[current_data_index:current_data_index+batch_size]
        current_label= train_label[current_data_index:current_data_index+batch_size]

        layer0,layer0_cache = conv_forward(current_data,l0_w,l0_b,padding=1)
        layer0_act = np_relu(layer0)
        layer0_pool,layer0_pool_cache = maxpool_forward(layer0_act)
        layer0_reshape_1 = layer0_pool.reshape(batch_size,-1)
        layer0_special = l0_special.feedforward(layer0_reshape_1.T).T
        layer0_reshape_2 = layer0_special.reshape(batch_size,layer0_output_c,14,14)

        layer1,layer1_cache = conv_forward(layer0_reshape_2,l1_w,l1_b,padding=1)
        layer1_act = np_relu(layer1)
        layer1_pool,layer1_pool_cache = maxpool_forward(layer1_act)
        layer1_reshape_1 = layer1_pool.reshape(batch_size,-1) #
        layer2 = l2.feedforward(layer1_reshape_1) #

        cost = np.mean( layer2 - current_label )
        accuracy = np.mean(np.argmax(layer2,1) == np.argmax(current_label, 1))
        print('Current Iter: ', iter,' batch index: ', current_data_index, ' accuracy: ',accuracy, ' cost: ',cost,end='\r')
        train_cota = train_cota + cost; train_acca = train_acca + accuracy

        grad2 = l2.backprop(layer2 - current_label ,lr_rate=learning_rate)
        grad2_reshape = grad2.reshape(batch_size,layer1_output_c,7,7)
        grad1_back_pool = maxpool_backward(grad2_reshape,layer1_pool_cache)
        grad1_back_act  = d_np_relu(grad1_back_pool)
        grad1_x,grad1_w,grad1_b = conv_backward(grad1_back_act,layer1_cache)

        grad0_reshape_1 = grad1_x.reshape(batch_size,-1)
        grad0_special = l0_special.backprop(grad0_reshape_1.T).T
        grad0_reshape_2 = grad0_special.reshape(batch_size,layer0_output_c,14,14)
        grad0_back_pool = maxpool_backward(grad0_reshape_2,layer0_pool_cache)
        grad0_back_act  = d_np_relu(grad0_back_pool)
        grad0_x,grad0_w,grad0_b = conv_backward(grad0_back_act,layer0_cache)

        # ===== update weights =====
        # layer 1 reg and adam ====
        grad1_w = grad1_w + lamda * l1_w
        grad1_b = grad1_b + lamda * l1_b

        l1_w_m = l1_w_m * beta1 + (1.-beta1) * grad1_w
        l1_w_v = l1_w_v * beta1 + (1.-beta1) * grad1_w ** 2
        m_hat,v_hat = l1_w_m/(1.-beta1), l1_w_v/(1.-beta2)
        adam_middle =  m_hat * learning_rate / (np.sqrt(v_hat) + adam_e)
        l1_w = l1_w - adam_middle

        l1_b_m = l1_b_m * beta1 + (1.-beta1) * grad1_b
        l1_b_v = l1_b_v * beta1 + (1.-beta1) * grad1_b ** 2
        m_hat,v_hat = l1_b_m/(1.-beta1), l1_b_v/(1.-beta2)
        adam_middle =  m_hat * learning_rate / (np.sqrt(v_hat) + adam_e)
        l1_b = l1_b - adam_middle
        # layer 1 reg and adam ====

        # layer 0 reg and adam ====
        grad0_w = grad0_w + lamda * l0_w
        grad0_b = grad0_b + lamda * l0_b

        l0_w_m = l0_w_m * beta1 + (1.-beta1) * grad0_w
        l0_w_v = l0_w_v * beta1 + (1.-beta1) * grad0_w ** 2
        m_hat,v_hat = l0_w_m/(1.-beta1), l0_w_v/(1.-beta2)
        adam_middle =  m_hat * learning_rate / (np.sqrt(v_hat) + adam_e)
        l0_w = l0_w - adam_middle

        l0_b_m = l0_b_m * beta1 + (1.-beta1) * grad0_b
        l0_b_v = l0_b_v * beta1 + (1.-beta1) * grad0_b ** 2
        m_hat,v_hat = l0_b_m/(1.-beta1), l0_b_v/(1.-beta2)
        adam_middle =  m_hat * learning_rate / (np.sqrt(v_hat) + adam_e)
        l0_b = l0_b - adam_middle
        # layer 1 reg and adam ====
        # ===== update weights =====

    for current_data_index in range(0,len(test_data),batch_size):
        current_data = test_data[current_data_index:current_data_index+batch_size]
        current_label= test_label[current_data_index:current_data_index+batch_size]

        layer0,layer0_cache = conv_forward(current_data,l0_w,l0_b,padding=1)
        layer0_act = np_relu(layer0)
        layer0_pool,layer0_pool_cache = maxpool_forward(layer0_act)
        layer0_reshape_1 = layer0_pool.reshape(batch_size,-1)
        layer0_special = l0_special.feedforward(layer0_reshape_1.T).T
        layer0_reshape_2 = layer0_special.reshape(batch_size,layer0_output_c,14,14)

        layer1,layer1_cache = conv_forward(layer0_reshape_2,l1_w,l1_b,padding=1)
        layer1_act = np_relu(layer1)
        layer1_pool,layer1_pool_cache = maxpool_forward(layer1_act)
        layer1_reshape_1 = layer1_pool.reshape(batch_size,-1) #
        layer2 = l2.feedforward(layer1_reshape_1) #

        cost = np.mean( layer2 - current_label )
        accuracy = np.mean(np.argmax(layer2,1) == np.argmax(current_label, 1))
        print('Current Iter: ', iter,' batch index: ', current_data_index, ' accuracy: ',accuracy, ' cost: ',cost,end='\r')
        test_cota = test_cota + cost; test_acca = test_acca + accuracy

    if iter % print_size==0:
        print("\n----------")
        print('Train Current Acc: ', train_acca/(len(train_data)/batch_size),' Current cost: ', train_cota/(len(train_data)/batch_size),end='\n')
        print('Test  Current Acc: ', test_acca/(len(test_data)/batch_size),' Current cost: ', test_cota/(len(test_data)/batch_size),end='\n')
        print("----------")

    train_acc.append(train_acca/(len(train_data)/batch_size))
    train_cot.append(train_cota/(len(train_data)/batch_size))
    train_cota,train_acca = 0,0
    test_acc.append(test_acca/(len(test_data)/batch_size))
    test_cot.append(test_cota/(len(test_data)/batch_size))
    test_cota,test_acca = 0,0


# -- end code --
