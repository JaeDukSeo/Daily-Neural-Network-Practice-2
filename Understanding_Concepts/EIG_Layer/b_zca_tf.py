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
tf.set_random_seed(6728)
np.random.seed(0)
np.set_printoptions(precision = 3,suppress =True)
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

# ======= Activation Function  ==========
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float64)  + (tf_elu(tf.cast(tf.less_equal(x,0),tf.float64) * x) + 1.0)

def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1 - tf_tanh(x) ** 2

def tf_sigmoid(x): return tf.nn.sigmoid(x)
def d_tf_sigmoid(x): return tf_sigmoid(x) * (1.0-tf_sigmoid(x))

def tf_atan(x): return tf.atan(x)
def d_tf_atan(x): return 1.0/(1.0 + x**2)

def tf_iden(x): return x
def d_tf_iden(x): return 1.0

def tf_softmax(x): return tf.nn.softmax(x)
def softabs(x): return tf.sqrt(x ** 2 + 1e-20)
# ======= Activation Function  ==========

# Func: Fully Connected Layer
class FNN():

    def __init__(self,inc,outc,act,d_act,special_init=False):
        if special_init:
            interval = np.sqrt(6.0 / (inc + outc + 1.0))
            self.w  = tf.Variable(tf.random_uniform(shape=(inc, outc),minval=-interval,maxval=interval,dtype=tf.float64,seed=4))
        else:
            self.w = tf.Variable(tf.random_normal([inc,outc], stddev=0.05,seed=2,dtype=tf.float64))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act

    def getw(self): return self.w

    def feedforward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient=None,l2_regularization=True):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

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
        adam_middle = m_hat *  learning_rate/(tf.sqrt(v_hat) + adam_e)

        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middle )))
        return grad_pass,update_w

# Def: ZCA white layer
class zca_whiten_layer():

    def __init__(self): pass

    def feedforward(self,input,EPS=10e-5):
        self.input = input
        self.sigma = tf.matmul(tf.transpose(input),input) / batch_size
        self.eigenval,self.eigvector = tf.linalg.eigh(self.sigma)
        self.U = tf.matmul(tf.matmul(self.eigvector,tf.diag(1./ tf.sqrt(self.eigenval+EPS))),tf.transpose(self.eigvector))
        self.whiten = tf.matmul(input,self.U)
        return self.whiten

    def backprop(self,grad,EPS=10e-5):
        d_U = tf.matmul(tf.transpose(self.input),grad)

        # ===== tf =====
        d_eig_value = self.eigvector.T.dot(d_U).dot(self.eigvector) * (-0.5) * np.diag(1. / (self.eigenval+EPS) ** 1.5)
        d_eig_vector = d_U.dot( (np.diag(1. / np.sqrt(self.eigenval+EPS)).dot(self.eigvector.T)).T  ) + (self.eigvector.dot(np.diag(1. / np.sqrt(self.eigenval+EPS)))).dot(d_U)
        E = np.ones((grad.shape[1],1)).dot(np.expand_dims(self.eigenval.T,0)) - np.expand_dims(self.eigenval,1).dot(np.ones((1,grad.shape[1])))
        K_matrix = 1./(E + np.eye(grad.shape[1])) - np.eye(grad.shape[1])
        np.fill_diagonal(d_eig_value,0.0)
        d_sigma = self.eigvector.dot(
                    K_matrix.T * (self.eigvector.T.dot(d_eig_vector)) + d_eig_value
                    ).dot(self.eigvector.T)
        d_x = grad.dot(self.U.T) + (2./grad.shape[0]) * self.input.dot(d_sigma) * 2
        # ===== tf =====

        return d_x

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

# hyper
num_epoch = 100
batch_size = 1250
learning_rate = 0.0002
lamda = 0.000008
print_size  = 1
beta1,beta2,adam_e = 0.9,0.999,1e-20

# class of layers
l0 = FNN(28*28,36*36, act=tf_tanh,d_act=d_tf_tanh)
l0_special = zca_whiten_layer()
l1 = FNN(36*36,42*42 , act=tf_tanh,d_act=d_tf_tanh)
l2 = FNN(42*42,10    , act=tf_sigmoid,d_act=d_tf_sigmoid)

# graph
x = tf.placeholder(shape=[batch_size,784],dtype=tf.float64)
y = tf.placeholder(shape=[batch_size,10],dtype=tf.float64)

layer0 = l0.feedforward(x)
layer0_special = l0_special.feedforward(layer0)
layer1 = l1.feedforward(layer0_special)
layer2 = l2.feedforward(layer1)

cost = tf.reduce_mean( tf.square(layer2 - y ))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layer2,1),tf.argmax(y, 1)),"float"))
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# train
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):

        train_cota,train_acca = 0,0;train_cot,train_acc = [],[]
        test_cota,test_acca = 0,0;test_cot,test_acc = [],[]
        train_data,train_label = shuffle(train_data,train_label)

        for current_data_index in range(0,len(train_data),batch_size):
            current_data = train_data[current_data_index:current_data_index+batch_size].astype(np.float64)
            current_label= train_label[current_data_index:current_data_index+batch_size].astype(np.float64)
            sess_results = sess.run([cost,accuracy,auto_train],feed_dict={x:current_data,y:current_label})
            print('Current Iter: ', iter,' batch index: ', current_data_index, ' accuracy: ',sess_results[1], ' cost: ',sess_results[0],end='\r')
            train_cota = train_cota + sess_results[0]; train_acca = train_acca + sess_results[1]

        for current_data_index in range(0,len(test_data),batch_size):
            current_data = test_data[current_data_index:current_data_index+batch_size].astype(np.float64)
            current_label= test_label[current_data_index:current_data_index+batch_size].astype(np.float64)
            sess_results = sess.run([cost,accuracy],feed_dict={x:current_data,y:current_label})
            print('Current Iter: ', iter,' batch index: ', current_data_index, ' accuracy: ',sess_results[1], ' cost: ',sess_results[0],end='\r')
            test_cota = test_cota + sess_results[0]; test_acca = test_acca + sess_results[1]

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
