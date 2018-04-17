import tensorflow as tf
import numpy as np
import sys, os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt

np.random.seed(6789)
tf.set_random_seed(678)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)

# read data
data_location = "../Dataset/Semanticdataset100/image/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            train_data.append(os.path.join(dirName,filename))

data_location = "../Dataset/Semanticdataset100/ground-truth/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".png" in filename.lower():  # check whether the file's DICOM
            train_data_gt.append(os.path.join(dirName,filename))


train_images = np.zeros(shape=(100,256,256,1))
train_labels = np.zeros(shape=(100,256,256,1))

for file_index in range(len(train_data)):
    train_images[file_index,:,:]   = np.expand_dims(imresize(imread(train_data[file_index],mode='F',flatten=True),(256,256)),axis=2)
    train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(256,256)),axis=2)

train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min())

# make class
class reslayer():
    
    def __init__(self,k,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([k,k,in_c,out_c],stddev=0.0005))

    def feedforward(self,input):
        self.input = input
        self.layer = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerA = tf_relu(self.layer) + input
        return self.layerA

# hyper
num_epoch = 100
learning_rate = 0.0001
batch_size = 2
print_size = 5

# define class

# make graph 
x = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)





# make session
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())


# -- end code --