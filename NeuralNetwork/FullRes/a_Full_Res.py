import tensorflow as tf
import numpy as np
import sys, os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as     

np.random.seed(6789)
tf.set_random_seed(678)

def tf_log(x): return tf.nn.sigmoid(x)
def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)

# read data
data_location = "../Dataset/kaggleNerve/sub/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower() and not 'mask'in filename.lower() :  # check whether the file's DICOM
            train_data.append(os.path.join(dirName,filename))

data_location = "../Dataset/kaggleNerve/sub/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower() and  'mask'in filename.lower():  # check whether the file's DICOM
            train_data_gt.append(os.path.join(dirName,filename))


train_images = np.zeros(shape=(64,256,256,1))
train_labels = np.zeros(shape=(64,256,256,1))

for file_index in range(len(train_data)):
    train_images[file_index,:,:]   = np.expand_dims(imresize(imread(train_data[file_index],mode='F',flatten=True),(256,256)),axis=2)
    train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(256,256)),axis=2)

train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min())

# make class
class reslayer():
    
    def __init__(self,k,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([k,k,in_c,out_c],stddev=0.05))

    def feedforward(self,input,add):
        self.input = input
        self.layer = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerA = tf_relu(self.layer) + add
        return self.layerA

class frrlayer():
    
    def __init__(self,k,in_c,out_c,up_c):
        self.w1 =tf.Variable(tf.random_normal([k,k,in_c,out_c],stddev=0.005))
        self.w2 =tf.Variable(tf.random_normal([k,k,out_c,out_c],stddev=0.005))

        self.wout = tf.Variable(tf.random_normal([1,1,out_c,up_c],stddev=0.005))

    def feedforward(self,input):
        self.input = input

        self.layer1 = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        self.layer1BN = tf.nn.batch_normalization(self.layer1,scale=True,offset=True,mean=0.0,variance=1.0,variance_epsilon=1e-8)
        self.layer1A = tf_relu(self.layer1BN) 

        self.layer2 = tf.nn.conv2d(self.layer1A ,self.w2,strides=[1,1,1,1],padding='SAME')
        self.layer2BN = tf.nn.batch_normalization(self.layer2,scale=True,offset=True,mean=0.0,variance=1.0,variance_epsilon=1e-8)
        self.layer2A = tf_relu(self.layer2BN) 

        self.layerout  = tf.nn.conv2d(self.layer2A ,self.wout,strides=[1,1,1,1],padding='SAME')

        return self.layer2A, self.layerout


# code from https://github.com/tensorflow/tensorflow/issues/8246
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

# hyper
num_epoch = 100
learning_rate = 0.0001
batch_size = 2
print_size = 5

# define class
l1_res = reslayer(3,1,1)

l2_res = reslayer(3,1,1)
l2_frr = frrlayer(3,1,1,1)

l3_res = reslayer(3,1,1)
l3_frr = frrlayer(3,2,1,1)

l4_res = reslayer(3,1,1)
l4_frr = frrlayer(3,2,1,1)

l5_res = reslayer(3,1,1)

# make graph 
x = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)

layer1_res = l1_res.feedforward(x,x)
layer1_DS_res = tf.nn.max_pool(layer1_res,ksize=[1,2,2,1], strides=[1, 2, 2,1], padding='SAME')

layer2_frr,layer2_out = l2_frr.feedforward(layer1_DS_res)
layer2_DS = tf.nn.max_pool(layer2_frr,ksize=[1,2,2,1], strides=[1, 2, 2,1], padding='SAME')
layer2_US_res = tf_repeat(layer2_out,[1,2,2,1])
layer2_res = l2_res.feedforward(layer1_res,layer2_US_res)
layer2_DS_res = tf.nn.max_pool(layer2_res,ksize=[1,4,4,1], strides=[1, 4, 4,1], padding='SAME')

layer3_Input = tf.concat([layer2_DS,layer2_DS_res],3)
layer3_frr,layer3_out = l3_frr.feedforward(layer3_Input)
layer3_US =  tf_repeat(layer3_out,[1,2,2,1])
layer3_US_res = tf_repeat(layer3_out,[1,4,4,1])
layer3_res = l3_res.feedforward(layer2_res,layer3_US_res)
layer3_DS_res = tf.nn.max_pool(layer3_res,ksize=[1,2,2,1], strides=[1, 2, 2,1], padding='SAME')

layer4_Input = tf.concat([layer3_US,layer3_DS_res],3)
layer4_frr,layer4_out = l4_frr.feedforward(layer4_Input)
layer4_US =  tf_repeat(layer4_out,[1,2,2,1])
layer4_US_res = tf_repeat(layer4_out,[1,2,2,1])
layer4_res = l4_res.feedforward(layer3_res,layer4_US_res)

layer5_res = l5_res.feedforward(layer4_res,layer4_res)

cost = tf.reduce_mean(tf.square(layer5_res-y))
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# make session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        for current_batch_index in range(0,len(train_images),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([cost,auto_train,layer5_res],feed_dict={x:current_batch,y:current_label})
            print(' Iter: ', iter, " Cost:  %.32f"% sess_results[0],end='\r')
        print('\n-----------------------')
        train_images,train_labels = shuffle(train_images,train_labels)

        if iter % 2 == 0:
            test_example =   train_images[:2,:,:,:]
            test_example_gt = train_labels[:2,:,:,:]
            sess_results = sess.run([layer5_res],feed_dict={x:test_example})

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
                sess_results = sess.run([cost,auto_train,layer5_res],feed_dict={x:current_batch,y:current_label})

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
                plt.imshow(np.squeeze(sess_results[2][0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"c_Generated Mask")
                plt.savefig('gif/'+str(current_batch_index)+"c_Generated_Mask.png")

                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"d_Original Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"d_Original_Image_Overlay.png")
            
                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[2][0,:,:,:])),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"e_Generated Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"e_Generated_Image_Overlay.png")

                plt.close('all')

# -- end code --