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
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.02))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
    def getw(self): return [self.w]

    def feedforward(self,input,stride=1,batch_norm=True,padding_val='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding=padding_val)
        if batch_norm: self.layer = tf.nn.batch_normalization(self.layer,mean=0.0,variance=1.0,variance_epsilon=1e-8,scale=True,offset=True)
        self.layer = tf.nn.avg_pool(self.layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
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

train_images = np.zeros(shape=(850,256,256,3))
train_labels = np.zeros(shape=(850,256,256,1))

for file_index in range(len(train_data)-1):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='RGB'),(256,256))
    train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(256,256)),axis=3)

train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0)+ 1e-10)
train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0)+ 1e-10)
train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0)+ 1e-10)
train_labels[:,:,:,0]  = (train_labels[:,:,:,0] - train_labels[:,:,:,0].min(axis=0)) / (train_labels[:,:,:,0].max(axis=0) - train_labels[:,:,:,0].min(axis=0)+ 1e-10)

# hyper
num_epoch = 100
learing_rate = 0.001
batch_size = 10

# define class
l1_e = CNNLayer(3,3,16,tf_Relu,d_tf_Relu)
l2_e = CNNLayer(3,16,32,tf_Relu,d_tf_Relu)
l3_e = CNNLayer(3,32,64,tf_Relu,d_tf_Relu)
l4_e = CNNLayer(3,64,1,tf_Relu,d_tf_Relu)

l1_match = CNNLayer(3,64,1,tf_Relu,d_tf_Relu)
l2_match = CNNLayer(3,32,1,tf_Relu,d_tf_Relu)
l3_match = CNNLayer(3,16,1,tf_Relu,d_tf_Relu)
l4_match = CNNLayer(3,1,1,tf_Relu,d_tf_Relu)

l1_d = CNNLayer(3,2,1,tf_Relu,d_tf_Relu)
l2_d = CNNLayer(3,2,1,tf_Relu,d_tf_Relu)
l3_d = CNNLayer(3,2,1,tf_Relu,d_tf_Relu)
l4_d = CNNLayer(3,2,1,tf_Relu,d_tf_Relu)

# graph
x = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)

y1 = tf.placeholder(shape=[None,8,8,1],dtype=tf.float32)
y2 = tf.placeholder(shape=[None,16,16,1],dtype=tf.float32)
y3 = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32)
y4 = tf.placeholder(shape=[None,64,64,1],dtype=tf.float32)
y5 = tf.placeholder(shape=[None,128,128,1],dtype=tf.float32)

# encoder portion
layer1 = l1_e.feedforward(x)
layer2 = l2_e.feedforward(layer1)
layer3 = l3_e.feedforward(layer2)
layer4 = l4_e.feedforward(layer3)
cost1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer4,labels=y1)

layer5_Match = l1_match.feedforward(layer4)
layer5_Input = tf.concat([layer4,layer5_Match],axis=3)
layer5 = l1_d.feedforward(layer5_Input)
layer5_Up = tf.image.resize_images(layer5,size=[16,16],method=tf.images.ResizeMethod.BILINEAR)
cost2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer5_Up,labels=y2)

# layer6_Match = l2_match.feedforward(layer5)
# layer6_Input = tf.concat([layer3,layer6_Match],axis=3)
# layer6 = l2_d.feedforward(layer6_Input)
# cost3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer6,label=y3)

# cost2 = tf.nn.softmax_cross_entropy_with_logits_v2(logit=layer4,label=y2)
# cost3 = tf.nn.softmax_cross_entropy_with_logits_v2(logit=layer4,label=y3)
# cost4 = tf.nn.softmax_cross_entropy_with_logits_v2(logit=layer4,label=y4)
# cost5 = tf.nn.softmax_cross_entropy_with_logits_v2(logit=layer4,label=y5)


# decoder portion





# session


# -- end code --