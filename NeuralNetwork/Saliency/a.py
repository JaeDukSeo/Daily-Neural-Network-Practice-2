import tensorflow as tf
import numpy as np
import sys, os
import cv2
from sklearn.utils import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
from skimage.transform import resize

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(x): return tf.cast(tf.greater(x,0),tf.float32)
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32) + tf_elu(tf.cast(tf.less_equal(x,0),tf.float32)*x)

np.random.seed(676)
tf.set_random_seed(6787)

# data
data_location = "../../Dataset/Semanticdataset100/image/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".jpg" in filename.lower():
            train_data.append(os.path.join(dirName,filename))

data_location = "../../Dataset/Semanticdataset100/ground-truth/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".png" in filename.lower() :
            train_data_gt.append(os.path.join(dirName,filename))

train_images = np.zeros(shape=(850,256,256,3))
train_labels = np.zeros(shape=(850,256,256,1))

for file_index in range(len(train_data)):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index]),(256,256))
    train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(256,256)),axis=2)

train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0))
train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0))
train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0))
train_labels = (train_labels - train_labels.min(axis=0)) / (train_labels.max(axis=0) - train_labels.min(axis=0))

# class
class cnn():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.005))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,maxpool=True):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layer = tf_relu(self.layer) 
        if maxpool: return self.layer
        else: return tf.nn.max_pool(self.layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    def backprop(self,gradient):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_relu(self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
            filter= self.w,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))
        
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)

        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,grad_update     

class DeCNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.005))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,de_stride = 2,shape=2):
        self.input  = input
        output_shape = self.input.shape[2].value * shape
        self.layer = tf.nn.conv2d_transpose(input,self.w,output_shape=[batch_size,output_shape,output_shape,60],strides=[1,de_stride,de_stride,1],padding='SAME')
        return self.layer

    def backprop(self,gradient):
        half_shape = gradient.shape[1].value//2
        grad_part_1 = gradient 
        grad_part_3 = self.input

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_part_1,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
            filter= self.w,out_backprop = grad_part_1,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))
        
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)

        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,grad_update     

# hyper
num_epoch = 100
batch_size = 50
learning_rate = 0.000001

# define class
l1a = cnn(3,3,64)
l1b = cnn(3,64,128)

l2a = cnn(3,128,128)
l2b = cnn(3,128,256)

l3a = cnn(3,256,256)
l3b = cnn(3,256,256)
l3c = cnn(3,256,512)

l4a = cnn(3,512,512)
l4b = cnn(3,512,512)
l4c = cnn(3,512,512)

l5a = cnn(3,512,512)
l5b = cnn(3,512,512)
l5c = cnn(3,512,512)

l6 = cnn(7,512,4096)
l7 = cnn(1,4096,4096)

l8a = DeCNN(4,60,4096)
l8b = cnn(1,60,60)

l9a = DeCNN(4,60,60)
l9b = cnn(3,60,60)

l10 = DeCNN(16,60,60)

l11a = cnn(1,60,12)
l11b = cnn(1,12,1)

# graph
x = tf.placeholder(shape=[None,256,256,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)

layer1a = l1a.feedforward(x)
layer1b = l1b.feedforward(layer1a,False) 

layer2a = l2a.feedforward(layer1b) 
layer2b = l2b.feedforward(layer2a,False) 

layer3a = l3a.feedforward(layer2b) 
layer3b = l3b.feedforward(layer3a) 
layer3c = l3c.feedforward(layer3b,False) 

layer4a = l4a.feedforward(layer3c) 
layer4b = l4b.feedforward(layer4a) 
layer4c = l4c.feedforward(layer4b,False) 

layer5a = l5a.feedforward(layer4c) 
layer5b = l5b.feedforward(layer5a) 
layer5c = l5c.feedforward(layer5b,False) 

layer6 = tf.nn.dropout(l6.feedforward(layer5c),0.5)
layer7 = tf.nn.dropout(l7.feedforward(layer6),0.5)

layer8a = l8a.feedforward(layer7)
layer8b = l8b.feedforward(layer8a)

layer9a = l9a.feedforward(layer8b)
layer9b = l9b.feedforward(layer9a)

layer10 = l10.feedforward(layer9b,de_stride=8,shape=8)

layer11a = l11a.feedforward(layer10)
layer11b = l11b.feedforward(layer11a)

cost = tf.reduce_mean(tf.square(layer11b-y) * 0.5)

# -- auto train --
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# session
with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        for current_batch_index in range(0,len(train_images),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_label})
            print(' Iter: ', iter, " Cost:  %.32f"% sess_results[0],end='\r')
        print('\n-----------------------')
        train_images,train_labels = shuffle(train_images,train_labels)

        if iter % 2 == 0:
            test_example =   train_images[:batch_size,:,:,:]
            test_example_gt = train_labels[:batch_size,:,:,:]
            sess_results = sess.run([layer11b],feed_dict={x:test_example})

            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]
            test_example_gt = test_example_gt[0,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+'Original Image')
            plt.savefig('train_change/epoch_'+str(iter)+"a_Original_Image.png")

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+'Ground Truth Mask')
            plt.savefig('train_change/epoch_'+str(iter)+"b_Original_Mask.png")

            plt.figure()
            plt.imshow(np.squeeze(sess_results),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+'Generated Mask')
            plt.savefig('train_change/epoch_'+str(iter)+"c_Generated_Mask.png")

            plt.figure()
            plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(test_example_gt)),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+"Ground Truth Overlay")
            plt.savefig('train_change/epoch_'+str(iter)+"d_Original_Image_Overlay.png")

            plt.figure()
            plt.axis('off')
            plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(sess_results)),cmap='gray')
            plt.title('epoch_'+str(iter)+"Generated Overlay")
            plt.savefig('train_change/epoch_'+str(iter)+"e_Generated_Image_Overlay.png")

            plt.close('all')

        # save image if it is last epoch
        if iter == num_epoch - 1:
            train_images,train_labels = shuffle(train_images,train_labels)
            
            for current_batch_index in range(0,len(train_images),batch_size):
                current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
                current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
                sess_results = sess.run([layer5_res],feed_dict={x:current_batch,y:current_label})

                plt.figure()
                plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"a_Original Image")
                plt.savefig('gif/'+str(current_batch_index)+"a_Original_Image.png")

                plt.figure()
                plt.imshow(np.squeeze(current_label[0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"b_Original Mask")
                plt.savefig('gif/'+str(current_batch_index)+"b_Original_Mask.png")
                
                plt.figure()
                plt.imshow(np.squeeze(sess_results[0][0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"c_Generated Mask")
                plt.savefig('gif/'+str(current_batch_index)+"c_Generated_Mask.png")

                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"d_Original Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"d_Original_Image_Overlay.png")
            
                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[0][0,:,:,:])),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"e_Generated Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"e_Generated_Image_Overlay.png")

                plt.close('all')
# -- end code --