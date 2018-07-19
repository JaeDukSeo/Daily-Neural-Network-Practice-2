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
from skimage.color import rgba2rgb

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(6278)
tf.set_random_seed(6728)
ia.seed(6278)

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
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05))
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
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05))
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
        self.w = tf.Variable(tf.random_normal([input_dim,hidden_dim], stddev=0.05))
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

class SOM_Layer(): 

    def __init__(self,m,n,dim,num_epoch,learning_rate_som = 0.04,radius_factor = 1.1,
    gaussian_std=0.5):
        
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

        self.neighbourhood_func = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, "float32")), tf.multiply(
                tf.square(tf.multiply(radius, self.gaussian_std)), 2)))

        self.learning_rate_op = tf.multiply(self.neighbourhood_func, alpha)
        
        self.numerator = tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.learning_rate_op, axis=-1),
            tf.expand_dims(self.input, axis=1)), axis=0)

        self.denominator = tf.expand_dims(
            tf.reduce_sum(self.learning_rate_op,axis=0) + float(1e-20), axis=-1)

        self.new_weights = tf.div(self.numerator, self.denominator)
        self.update = [tf.assign(self.map, self.new_weights)]

        return self.update,tf.reduce_mean(self.grad_pass, 1)

# ================= LAYER CLASSES =================

# data
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
train, test = tf.keras.datasets.mnist.load_data()
x_data, train_label, y_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

train_batch = x_data/255.0
test_batch = y_data/255.0
train_batch = train_batch[:3000,:]
train_label = train_label[:3000,:]
test_batch = test_batch[:100,:]
test_label = test_label[:100,:]

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

train_batch = train_batch/255.0
test_batch = test_batch/255.0
train_batch = train_batch[:100]
train_label = train_label[:100]
test_batch = test_batch[:100]
test_label = test_label[:100]

# hyper parameter 10000
num_epoch = 200
batch_size = 100
print_size = 2

map_width_height  = 50
map_dim = 3

learning_rate = 0.0001
learnind_rate_decay = 0.0
beta1,beta2,adam_e = 0.9,0.999,1e-8

# define class here
el1 = FNN(784,64,act=tf_atan,d_act=d_tf_atan)
el2 = FNN(64,3,act=tf_atan,d_act=d_tf_atan)
SOM_layer = SOM_Layer(map_width_height,map_width_height,map_dim,num_epoch = num_epoch,
learning_rate_som=0.9,radius_factor=1.1,gaussian_std = 0.08 )

# graph
x = tf.placeholder(shape=[batch_size,784],dtype=tf.float32,name="input")
current_iter = tf.placeholder(shape=[],dtype=tf.float32)

# encoder
elayer1 = el1.feedforward(x)
elayer2 = el2.feedforward(elayer1)
SOM_layer.feedforward(elayer2)

SOM_Update,SOM_grad = SOM_layer.backprop(current_iter,num_epoch)
egrad2,egrad2_up = el2.backprop(SOM_grad)
egrad1,egrad1_up = el1.backprop(egrad2)

grad_update = SOM_Update + egrad2_up + egrad1_up

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # start the training
    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)
        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            sess_results = sess.run(grad_update,feed_dict={x:current_batch,current_iter:iter})
            print('Current Iter: ',iter,' Current Train Index: ',batch_size_index,
            ' Current SUM of updated Values: ',sess_results[0].sum(),end='\r' )
        print('\n-----------------------')


    # get the trained map and normalize
    trained_map = sess.run(SOM_layer.getmap()).reshape(map_width_height,map_width_height,3)
    trained_map[:,:,0] = (trained_map[:,:,0]-trained_map[:,:,0].min())/(trained_map[:,:,0].max()-trained_map[:,:,0].min())
    trained_map[:,:,1] = (trained_map[:,:,1]-trained_map[:,:,1].min())/(trained_map[:,:,1].max()-trained_map[:,:,1].min())
    trained_map[:,:,2] = (trained_map[:,:,2]-trained_map[:,:,2].min())/(trained_map[:,:,2].max()-trained_map[:,:,2].min())
    plt.imshow(trained_map)
    plt.show()

    # after training is done get the cloest vector
    locations = sess.run(SOM_layer.getlocation(),feed_dict={x:train_batch[:batch_size]})
    x1 = locations[:,0]; y1 = locations[:,1]
    index = [ np.where(r==1)[0][0] for r in train_label ]
    index = list(map(str, index))

    ## Plots: 1) Train 2) Test+Train ###
    plt.figure(1, figsize=(12,6))
    plt.subplot(121)
    plt.imshow(trained_map)
    plt.scatter(x1,y1)
    # Just adding text
    for i, m in enumerate(locations):
        plt.text( m[0], m[1],index[i], ha='center', va='center', 
        bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.title('Train MNIST 100')

    locations2 = sess.run(SOM_layer.getlocation(),feed_dict={x:test_batch})
    x2 = locations2[:,0]; y2 = locations2[:,1]
    index2 = [ np.where(r==1)[0][0] for r in test_label ]
    index2 = list(map(str, index2))

    plt.subplot(122)
    # Plot 2: Training + Testing
    plt.imshow(trained_map)
    plt.scatter(x1,y1)
    # Just adding text
    for i, m in enumerate(locations):
        plt.text( m[0], m[1],index[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.scatter(x2,y2)
    # Just adding text
    for i, m in enumerate(locations2):
        plt.text( m[0], m[1],index2[i], ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5, lw=0))
    plt.title('Test MNIST 10 + Train MNIST 100')
    plt.show()

# -- end code --