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

# create layer
def np_sigmoid(x): return 1.0 / (1.0+np.exp(-x))
def d_np_sigmoid(x): return np_sigmoid(x) * (1.0 - np_sigmoid(x))

# def: soft max function for 2D
def stable_softmax(x,axis=None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=1)[:,np.newaxis])
    return e_x / e_x.sum(axis=1)[:,np.newaxis]

# def: fully connected layer
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

# ===== SIMPLE MODULAR APPROACH =======
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
        dem = 1./(self.m * np.sqrt(self.std + EPS ) )
        d_x = self.m * grad - np.sum(grad,axis = 0) - self.x_hat*np.sum(grad*self.x_hat, axis=0)
        return d_x * dem

# def: zca whitening layer
class zca_whiten_layer():

    def __init__(self): pass

    def feedforward(self,input,EPS=10e-5):
        self.sigma = input.T.dot(input) / input.shape[0]
        self.eigenval,self.eigvector = np.linalg.eigh(self.sigma)
        self.U = self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS))).dot(self.eigvector.T)
        self.whiten = input.dot(self.U)
        return self.whiten

    def backprop(self):
        pass
# ===== SIMPLE MODULAR APPROACH =======

# ===== FULL ======
# def: Batch Normalization
class Batch_Normalization_layer():

    def __init__(self):
        pass

    def feedforward(self,input,EPS=10e-5):
        self.input = input
        self.mean  = np.sum(input,axis=0) / input.shape[0]
        self.std   = np.sum( (self.input - self.mean) ** 2,0)  / input.shape[0]
        self.x_hat = (input - self.mean) / np.sqrt(self.std + EPS)
        return self.x_hat

    def backprop(self,grad,EPS=10e-5):
        dem = 1./(self.m * np.sqrt(self.std + EPS ) )
        d_x = self.m * grad - np.sum(grad,axis = 0) - self.x_hat*np.sum(grad*self.x_hat, axis=0)
        return d_x * dem

# def: Decorrelated Batch Normalization
class Decorrelated_Batch_Norm():

    def __init__(self,batch_size,feature_size):
        self.m = batch_size
        self.n = feature_size

    def feedforward(self,input,EPS=1e-5):
        self.input = input
        self.mean = (1./self.m) * np.sum(input,axis=0)
        # self.sigma = (1./self.m) * (input - self.mean).T.dot(input - self.mean)
        self.sigma = np.cov(input-self.mean,ddof=0,rowvar=False)
        self.eigenval,self.eigvector = np.linalg.eigh(self.sigma)
        self.U = self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS))).dot(self.eigvector.T)
        self.whiten = (self.input-self.mean).dot(self.U)
        return self.whiten

    def backprop(self,grad,EPS=1e-5):

        # d_U = (self.input-self.mean).T.dot(grad)
        d_U = np.sum((self.input-self.mean),axis=0)[np.newaxis,:].T.dot(np.sum(grad,axis=0)[np.newaxis,:])

        d_eig_value  = self.eigvector.T.dot(d_U).dot(self.eigvector) * (-0.5) * np.diag(1. / (self.eigenval+EPS) ** 1.5)
        d_eig_vector =d_U.dot( (np.diag(1. / np.sqrt(self.eigenval+EPS)).dot(self.eigvector.T)).T  ) + \
                      self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS))).dot(d_U)

        E = np.ones((self.n,1)).dot(np.expand_dims(self.eigenval.T,0)) - \
            np.expand_dims(self.eigenval  ,1).dot(np.ones((1,self.n)))
        K_matrix = 1./(E + EPS) - np.eye(self.n)
        np.fill_diagonal(d_eig_value,0.0)

        d_sigma = self.eigvector.dot(
                    K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + d_eig_value
                    ).dot(self.eigvector.T)
        d_mean =  (-1)*np.sum(grad,0).dot(self.U.T) + (-2./self.m) * np.sum((self.input-self.mean),0).dot(d_sigma) * 2

        d_x = grad.dot(self.U.T) + (1./self.m) * d_mean + (2./self.m) * (self.input-self.mean).dot(d_sigma) * 2

        # ========= ========
        # # Paper Approach Sum and Dot Product
        # d_U = np.sum(self.input-self.mean,axis=0)[np.newaxis,:].T.dot(np.sum(grad,axis=0)[np.newaxis,:])
        #
        # d_eig_value = self.eigvector.T.dot(d_U) * (-1/2) * np.diag(1. / (self.eigenval+EPS) ** 1.5 )
        # # d_eig_vector = d_U.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)).T) + self.whiten.T.dot(grad)
        # # Paper Approach Sum and Dot Product
        # d_eig_vector = d_U.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)).T) +
        # np.sum(self.whiten,0)[np.newaxis,:].T.dot( np.sum(grad,0)[np.newaxis,:] )

        # E = np.ones((self.n,1)).dot(np.expand_dims(self.eigenval.T,0)) - \
        #     np.expand_dims(self.eigenval  ,1).dot(np.ones((1,self.n)))
        # K_matrix = 1./(E + np.eye(self.n)) - np.eye(self.n)
        # np.fill_diagonal(d_eig_value,0.0)
        #
        # d_sigma = self.eigvector.dot(
        #             K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + d_eig_value
        #             ).dot(self.eigvector.T)
        # d_simg_sym = (0.5) * (d_sigma.T + d_sigma)
        #
        # # d_mean = np.sum(grad.dot(self.U.T) * -1.0,0) + (-2./self.m) * np.sum( (self.input - self.mean).dot(d_simg_sym), 0  )
        # # Paper Approach Sum and Dot Product
        # d_mean = np.sum(grad,0).dot(self.U.T) * -1.0 + (-2./self.m) * np.sum( (self.input - self.mean), 0).dot(d_simg_sym)
        #
        # d_x = grad.dot(self.U.T) + \
        #       (2./self.m) * (self.input - self.mean).dot(d_simg_sym) + \
        #       (1./self.m) * d_mean
        # ========= ========

        return d_x
# ===== FULL ======

def simple_scale(x):
    return (x-x.min())/(x.max()-x.min())


# -- end code --
