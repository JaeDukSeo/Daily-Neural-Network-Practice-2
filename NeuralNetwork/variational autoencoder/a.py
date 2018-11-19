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

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(678)
tf.set_random_seed(678)
ia.seed(678)

def tf_elu(x): return x/(1+tf.abs(x))
def d_tf_elu(x): return 1/tf.square(1+tf.abs(x))

def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1 - tf_tanh(x) ** 2

def tf_sigmoid(x): return tf.nn.sigmoid(x) 
def d_tf_sigmoid(x): return tf_sigmoid(x) * (1.0-tf_sigmoid(x))

def tf_atan(x): return tf.atan(x)
def d_tf_atan(x): return 1.0/(1.0 + x**2)

def tf_softmax(x): return tf.nn.softmax(x)

# generating coordinates 
def coordinates(x_dim , y_dim , scale = 1.0):
    n_points = x_dim * y_dim
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), 1).reshape(n_points, 1)
    y_mat = np.tile(y_mat.flatten(), 1).reshape(n_points, 1)
    r_mat = np.tile(r_mat.flatten(), 1).reshape(n_points, 1)
    return x_mat, y_mat, r_mat

# class for the layers
class FNN():
    
    def __init__(self,input_dim,hidden_dim,act,d_act,stddev=1.0):
        self.w = tf.Variable(tf.random_normal([input_dim,hidden_dim], stddev=stddev))
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
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   ))
        v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 

        def f1(): return v_t
        def f2(): return self.v_hat_prev

        v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
        adam_middel = tf.multiply(learning_rate_change/(tf.sqrt(v_max) + adam_e),self.m)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel  )  ))
        update_w.append(tf.assign( self.v_prev,v_t ))
        update_w.append(tf.assign( self.v_hat_prev,v_max ))        
        return grad_pass,update_w   

# hyper
generated_image_x,generated_image_y = 1080,1080
generated_channel = 1
hidden_net_size = 64

number_of_image = 500

# class
z_layer = FNN(hidden_net_size,hidden_net_size,tf_atan,d_tf_tanh,stddev=1.0)
x_layer = FNN(1,hidden_net_size,tf_atan,d_tf_tanh,stddev=0.1)
y_layer = FNN(1,hidden_net_size,tf_tanh,d_tf_tanh,stddev=0.8)
r_layer = FNN(1,hidden_net_size,tf_atan,d_tf_tanh,stddev=1.0)

cl1 = FNN(hidden_net_size,hidden_net_size,tf_atan,d_tf_tanh)
cl2 = FNN(hidden_net_size,hidden_net_size,tf_atan,d_tf_tanh)
cl3 = FNN(hidden_net_size,hidden_net_size,tf_atan,d_tf_tanh)
cl4 = FNN(hidden_net_size,hidden_net_size,tf_atan,d_tf_tanh)
cl5 = FNN(hidden_net_size,generated_channel,tf_sigmoid,d_tf_sigmoid)

# graph
# latent vector, inputs to cppn, like coordinates and radius from centre
z = tf.placeholder(tf.float32, [None, hidden_net_size])
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])
r = tf.placeholder(tf.float32, [None, 1])

U_z = z_layer.feedforward(z)
U_x = x_layer.feedforward(x)
U_y = y_layer.feedforward(y)
U_r = r_layer.feedforward(r)

U = U_r + U_y + U_x + U_z

create_layer1 = cl1.feedforward(U)
create_layer2 = cl2.feedforward(create_layer1)
create_layer3 = cl3.feedforward(create_layer2)
create_layer4 = cl4.feedforward(create_layer3)
create_layer5 = cl5.feedforward(create_layer4)

generated_image = tf.reshape(create_layer5,[generated_image_x,generated_image_y,generated_channel])

# sesson
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for curren_image_number  in range(number_of_image):
        created_x,created_y,created_r = coordinates(generated_image_x,generated_image_y)
        # created_z = np.random.uniform(-1.0, 1.0, size=(created_x.shape[1],hidden_net_size )).astype(np.float32) * curren_image_number/number_of_image
        created_z = np.ones((created_x.shape[1],hidden_net_size )) * curren_image_number/number_of_image
        created_image = sess.run(generated_image,feed_dict={z:created_z,x:created_x,y:created_y,r:created_r})
       
        plt.axis('off')
        plt.imshow(np.squeeze(created_image))
        plt.savefig('viz/'+str(curren_image_number)+'.png',bbox_inches='tight')

# -- end code ---