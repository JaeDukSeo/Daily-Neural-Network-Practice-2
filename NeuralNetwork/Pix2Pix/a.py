import tensorflow as tf
import numpy as np,sys,os
from numpy import float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize

np.random.seed(678)
tf.set_random_seed(678)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),tf.float32)
def tf_LRelu(x): return tf.nn.leaky_relu(x)
def d_tf_LRelu(x): return tf.cast(tf.greater(x,0),tf.float32) + tf.cast(tf.less_equal(x,0),tf.float32) *x* 0.2
def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))
def tf_log(x): return tf.nn.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1-tf_log(x))

# convolution layer
class CNNLayer():
    
    def __init__(self,ker,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.005))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,stride=2,batch_norm=True,padding_val='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding=padding_val)
        if batch_norm: self.layer = tf.nn.batch_normalization(self.layer,mean=0.0,variance=1.0,variance_epsilon=1e-8,scale=True,offset=True)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient,stride=1):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,filter_sizes = self.w.shape,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        grad_pass  = tf.nn.conv2d_backprop_input(
            input_sizes=[batch_size] + list(self.input.shape[1:]),filter = self.w ,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        update_w = []

        update_w.append(
            tf.assign( self.m,self.m*beta_1 + (1-beta_1) * grad   )
        )
        update_w.append(
            tf.assign( self.v,self.v*beta_2 + (1-beta_2) * grad ** 2   )
        )

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,update_w

class CNNLayer_Up():
    
    def __init__(self,ker,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.005))
        self.act,self.d_act = act,d_act

    def feedforward(self,input,stride=1,dilate=1,output=1,batch_norm=True,dropout=False):
        self.input  = input
        current_shape_size = input.shape
        self.layer = tf.nn.conv2d_transpose(self.input,self.w,
        output_shape=[batch_size] + [int(current_shape_size[1].value * 2 ),int(current_shape_size[2].value*2 ),int(current_shape_size[3].value/int(self.w.shape[3].value/self.w.shape[2].value )  )],
        strides=[1,2,2,1],padding='SAME')
        if batch_norm: self.layer = tf.nn.batch_normalization(self.layer,mean=0.0,variance=1.0,variance_epsilon=1e-8,scale=True,offset=True)
        if dropout: self.layer = tf.nn.dropout( self.layer,0.5)
        self.layerA = self.act(self.layer)
        return self.layerA

# data
data_location = "../../Dataset/salObj/datasets/imgs/pascal/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".jpg" in filename.lower() :
            train_data.append(os.path.join(dirName,filename))

data_location =  "../../Dataset/salObj/datasets/masks/pascal/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".png" in filename.lower() :
            train_data_gt.append(os.path.join(dirName,filename))

train_images = np.zeros(shape=(850,255,255,3))
train_labels = np.zeros(shape=(850,255,255,1))

for file_index in range(len(train_data)-1):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='RGB'),(255,255))
    train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(255,255)),axis=3)

train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0)+ 1e-10)
train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0)+ 1e-10)
train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0)+ 1e-10)
train_labels[:,:,:,0]  = (train_labels[:,:,:,0] - train_labels[:,:,:,0].min(axis=0)) / (train_labels[:,:,:,0].max(axis=0) - train_labels[:,:,:,0].min(axis=0)+ 1e-10)

train_data = train_labels[50:,:,:,:]
train_gt =   train_images[50:,:,:,:]

test_data = train_labels[:50,:,:,:]
test_gt =   train_images[:50,:,:,:]

# hyper
num_epoch = 1000
learning_rate = 0.0001
batch_size = 5

beta1,beta2,adam_e = 0.9,0.999,1e-8

# generator ---------
# encoder (unet)
gl1 = CNNLayer(4,1,64,tf_LRelu,d_tf_LRelu)
gl2 = CNNLayer(4,64,128,tf_LRelu,d_tf_LRelu)
gl3 = CNNLayer(4,128,256,tf_LRelu,d_tf_LRelu)
gl4 = CNNLayer(4,256,512,tf_LRelu,d_tf_LRelu)
gl5 = CNNLayer(4,512,512,tf_LRelu,d_tf_LRelu)

# decoder
gl6 = CNNLayer_Up(4,512,512,tf_Relu,d_tf_Relu)
gl7 = CNNLayer_Up(4,256,1024,tf_Relu,d_tf_Relu)
gl8 = CNNLayer_Up(4,128,512,tf_Relu,d_tf_Relu)
gl9 = CNNLayer_Up(4,64,256,tf_Relu,d_tf_Relu)
gl10 = CNNLayer_Up(4,64,128,tf_Relu,d_tf_Relu)
glfinal = CNNLayer(4,64,3,tf_tanh,d_tf_tanh)
# generator ---------

# discrimator -------
dl1 = CNNLayer(4,4,64,tf_LRelu,d_tf_LRelu)
dl2 = CNNLayer(4,64,128,tf_LRelu,d_tf_LRelu)
dl3 = CNNLayer(4,128,256,tf_LRelu,d_tf_LRelu)
dl4 = CNNLayer(4,256,512,tf_LRelu,d_tf_LRelu)
dlfinal = CNNLayer(4,512,1,tf_log,d_tf_log)

# graph
input_binary_image = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)
color_image = tf.placeholder(shape=[None,256,256,3],dtype=tf.float32)

g_e_layer1 = gl1.feedforward(input_binary_image,batch_norm=False)
g_e_layer2 = gl2.feedforward(g_e_layer1)
g_e_layer3 = gl3.feedforward(g_e_layer2)
g_e_layer4 = gl4.feedforward(g_e_layer3)
g_e_layer5 = gl5.feedforward(g_e_layer4)

g_e_layer6 = gl6.feedforward(g_e_layer5)
g_e_layer7 = gl7.feedforward(tf.concat([g_e_layer6,g_e_layer4],axis=3))
g_e_layer8 = gl8.feedforward(tf.concat([g_e_layer7,g_e_layer3],axis=3))
g_e_layer9 = gl9.feedforward(tf.concat([g_e_layer8,g_e_layer2],axis=3))
g_e_layer10 = gl10.feedforward(tf.concat([g_e_layer9,g_e_layer1],axis=3))
g_e_layer_final= glfinal.feedforward(g_e_layer10,stride=1)

# -------- discriminator
true_discrim_input = tf.concat([input_binary_image,color_image],axis=3)
fake_discrim_input = tf.concat([input_binary_image,g_e_layer_final],axis=3)

true_d_1 = dl1.feedforward(true_discrim_input)
true_d_2 = dl2.feedforward(true_d_1)
true_d_3 = dl3.feedforward(true_d_2)
true_d_4 = dl4.feedforward(true_d_3)
true_d_f = dlfinal.feedforward(true_d_4)

fake_d_1 = dl1.feedforward(fake_discrim_input)
fake_d_2 = dl2.feedforward(fake_d_1)
fake_d_3 = dl3.feedforward(fake_d_2)
fake_d_4 = dl4.feedforward(fake_d_3)
fake_d_f = dlfinal.feedforward(fake_d_4)

# ----- losses
g_loss = tf.reduce_mean(-tf.log(fake_d_f)) + 100 * tf.reduce_mean(tf.abs(color_image - g_e_layer_final))
d_loss = tf.reduce_mean(-tf.log(true_d_f) + tf.log(1.0 - fake_d_f))

# --- training
auto_g_train = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5).minimize(g_loss)
auto_d_train = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5).minimize(d_loss)


# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        for current_batch_index in range(0,len(train_data),batch_size):
            current_image_batch = train_data[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_mask_batch  = train_gt[current_batch_index:current_batch_index+batch_size,:,:,:]

            
        

# -- end code --