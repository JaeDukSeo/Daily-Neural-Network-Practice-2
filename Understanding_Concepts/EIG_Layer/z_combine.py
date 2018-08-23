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

np.random.seed(0)
np.set_printoptions(precision = 3,suppress =True)
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

# import data
# mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
mnist = input_data.read_data_sets('../../Dataset/fashionmnist/', one_hot=True)
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

# soft max function for 2D
def stable_softmax(x,axis=None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=1)[:,np.newaxis])
    return e_x / e_x.sum(axis=1)[:,np.newaxis]

# fully connected layer
class np_FNN():

    def __init__(self,inc,outc,act=np_sigmoid,d_act = d_np_sigmoid):
        self.w = 1.0 * np.random.randn(inc,outc) + 0.0
        self.m,self.v = np.zeros_like(self.w),np.zeros_like(self.w)
        self.act = act; self.d_act = d_act

    def getw(self): return self.w
    def feedforward(self,input):
        self.input  = input
        self.layer  = self.input.dot(self.w)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,grad):
        grad_1 = grad
        grad_2 = self.d_act(self.layer)
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
        K_matrix = 1./(E + EPS) - np.eye(self.n)
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

    def feedforward(self,input,EPS=1e-5):
        self.input = input
        self.mean = (1./self.m) * np.sum(input,axis=0)
        self.sigma = (1./self.m) * (input - self.mean).T.dot(input - self.mean)
        self.eigenval,self.eigvector = np.linalg.eigh(self.sigma)
        self.U = self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS))).dot(self.eigvector.T)
        self.whiten = (input-self.mean).dot(self.U)
        return self.whiten

    def backprop(self,grad,EPS=1e-5):

        d_white = grad.dot(self.eigvector.T)

        d_U = (self.input-self.mean).T.dot(d_white)
        # Paper Approach Sum and Dot Product
        # d_U = np.sum(self.input-self.mean,axis=0)[np.newaxis,:].T.dot(np.sum(d_white,axis=0)[np.newaxis,:])

        d_eig_value = self.eigvector.T.dot(d_U) * (-1/2) * np.diag(1. / (self.eigenval+EPS) ** 1.5 )

        d_eig_vector = d_U.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)).T) + self.whiten.T.dot(grad)
        # Paper Approach Sum and Dot Product
        # d_eig_vector = d_U.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)).T) + \
        # np.sum(self.whiten,0)[np.newaxis,:].T.dot( np.sum(grad,0)[np.newaxis,:] )

        E = np.ones((self.n,1)).dot(np.expand_dims(self.eigenval.T,0)) - \
            np.expand_dims(self.eigenval  ,1).dot(np.ones((1,self.n)))
        K_matrix = 1./(E + np.eye(self.n)+EPS) - np.eye(self.n)

        np.fill_diagonal(d_eig_value,0.0)
        d_sigma = self.eigvector.dot(
                    K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + d_eig_value
                    ).dot(self.eigvector.T)
        d_simg_sym = (0.5) * (d_sigma.T + d_sigma)

        d_mean = np.sum(d_white.dot(self.U.T) * -1.0,0) + \
                 (-2./self.m) * np.sum( (self.input - self.mean).dot(d_simg_sym), 0  )

        # Paper Approach Sum and Dot Product
        # d_mean = np.sum(d_white,0).dot(self.U.T) * -1.0 + \
                 # (-2./self.m) * np.sum( (self.input - self.mean), 0).dot(d_simg_sym)

        d_x = d_white.dot(self.U.T) + \
              (2./self.m) * (self.input - self.mean).dot(d_simg_sym) + \
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
print_size = 1

learning_rate = 0.003
beta1,beta2,adam_e = 0.9,0.9,1e-8
small_batch_size = 100

# class
l0_test = Decorrelated_Batch_Norm(batch_size,784)
l0 = np_FNN(784,400)
l1 = Batch_Normalization_layer(batch_size,400)
l2 = np_FNN(400,300)
l3 = Decorrelated_Batch_Norm(batch_size,300)
l4 = np_FNN(300,100)
l5 = Batch_Normalization_layer(batch_size,100)
l6 = np_FNN(100,10)

def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert(X.ndim == 2)
    EPS = 10e-5

    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)
    # W = E.dot(D.dot(E.T))
    X_white = np.dot(X, W)

    # X_white = X.dot(np.dot(E, D))
    # X_white = X_white.dot(E)

    return X_white

# train
for iter in range(num_epoch):

    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    # train_data,train_label = shuffle(train_data,train_label)

    for current_batch_index in range(0,len(train_data),batch_size):

        current_train_data = train_data[current_batch_index:current_batch_index + batch_size]
        current_train_data_label = train_label[current_batch_index:current_batch_index + batch_size]

        # feed forward
        layer0 = l0.feedforward(current_train_data)
        layer1_full = l1.feedforward(layer0)

        layer2 = l2.feedforward(layer1_full)
        layer3_full = l3.feedforward(layer2)

        layer4 = l4.feedforward(layer3_full)
        layer5_full = l5.feedforward(layer4)

        layer6 = l6.feedforward(layer5_full)

        # cost
        final_soft = stable_softmax(layer6)
        cost = - np.mean(current_train_data_label * np.log(final_soft + 1e-20) + (1.0-current_train_data_label) * np.log(1.0-final_soft + 1e-20))
        correct_prediction = np.equal(np.argmax(final_soft, 1), np.argmax(current_train_data_label, 1))
        accuracy = np.mean(correct_prediction)
        print('Current Iter: ', iter,' batch index: ', current_batch_index, ' accuracy: ',accuracy, ' cost: ',cost,end='\r')
        train_cota = train_cota + cost
        train_acca = train_acca + accuracy

        # print('\n--------------------')
        # print( (final_soft-current_train_data_label).sum() )
        # print( (final_soft-current_train_data_label).mean() )
        # print( final_soft.sum(1) )
        # input()
        # print('--------------------\n')

        # back prop
        grad6 = l6.backprop(final_soft-current_train_data_label)

        grad5_full = l5.backprop(grad6)
        grad4 = l4.backprop(grad5_full)

        grad3_full = l3.backprop(grad4)
        grad2 = l2.backprop(grad3_full)

        grad1_full = l1.backprop(grad2)
        grad0 = l0.backprop(grad1_full)

    if iter % print_size==0:
        print("\n----------")
        print('Train Current Acc: ', train_acca/(len(train_data)/batch_size),' Current cost: ', train_cota/(len(train_data)/batch_size),end='\n')
        print("----------")

    train_acc.append(train_acca/(len(train_data)/batch_size))
    train_cot.append(train_cota/(len(train_data)/batch_size))
    train_cota,train_acca = 0,0







# ===== Compare ===
# layer0_test = l0_test.feedforward(current_train_data)
# # layer0_test = zca_whiten(current_train_data)
#
# # # compute the covariance of the image data
# mean_temp = np.mean(current_train_data,axis=0)
# cov = np.cov(current_train_data-mean_temp, rowvar=False)   # cov is (N, N)
# U,S,V = np.linalg.svd(cov)     # U is (N, N), S is (N,)
# epsilon = 1e-5
# zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
# zca = np.dot(current_train_data-mean_temp, zca_matrix)    # zca is (N, 3072)
#
# plt.subplot(1, 2, 1)
# plt.imshow(
# zca[0].reshape((28,28)),cmap='gray'
# )
# plt.title('not mine')
# plt.subplot(1, 2, 2)
# plt.imshow(
# layer0_test[0].reshape((28,28)),cmap='gray'
# )
# plt.title('mine')
# plt.show()
# sys.exit()
# ===== Compare ===





# -- end code --
