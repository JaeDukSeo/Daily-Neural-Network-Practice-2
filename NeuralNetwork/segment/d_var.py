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
from skimage.color import rgb2gray
import matplotlib

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
# ================= LAYER CLASSES =================

# data
data_location =  "../../Dataset/salObj/datasets/imgs/ft/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".jpg" in filename.lower() :
            train_data.append(os.path.join(dirName,filename))

data_location = "../../Dataset/salObj/datasets/masks/ft/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".png" in filename.lower() :
            train_data_gt.append(os.path.join(dirName,filename))

image_resize_px = 64
train_images = np.zeros(shape=(1000,image_resize_px,image_resize_px,3))
train_labels = np.zeros(shape=(1000,image_resize_px,image_resize_px,1))

for file_index in range(len(train_images)):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='RGB'),(image_resize_px,image_resize_px))
    train_labels[file_index,:,:]   = np.expand_dims(
        imresize(rgb2gray(imread(train_data_gt[file_index],mode='RGB')),(image_resize_px,image_resize_px)),3)
    
train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0)+1e-10)
train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0)+1e-10)
train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0)+1e-10)

train_labels[:,:,:,0]  = (train_labels[:,:,:,0] - train_labels[:,:,:,0].min(axis=0)) / (train_labels[:,:,:,0].max(axis=0) - train_labels[:,:,:,0].min(axis=0)+1e-10)
# train_labels[:,:,:,1]  = (train_labels[:,:,:,1] - train_labels[:,:,:,1].min(axis=0)) / (train_labels[:,:,:,1].max(axis=0) - train_labels[:,:,:,1].min(axis=0)+1e-10)
# train_labels[:,:,:,2]  = (train_labels[:,:,:,2] - train_labels[:,:,:,2].min(axis=0)) / (train_labels[:,:,:,2].max(axis=0) - train_labels[:,:,:,2].min(axis=0)+1e-10)

# split the data 
train_batch = train_images[:950]
train_label = train_labels[:950]
test_batch = train_images[950:]
test_label = train_labels[950:]

# train_batch = train_images[:50]
# train_label = train_labels[:50]
# test_batch = train_images[50:60]
# test_label = train_labels[50:60]

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

# hyper parameter 10000
num_epoch = 101
batch_size = 10
print_size = 2

learning_rate = 0.000003
learnind_rate_decay = 0.0
beta1,beta2,adam_e = 0.9,0.999,1e-8

# define class here
el1 = CNN(3,3,128)
el2 = CNN(3,128,256)
el3 = CNN(3,256,512)

e_mean = CNN(1,512,512)
e_var  = CNN(1,512,512)

dl1 = CNN_Trans(3,256,512)
dl2 = CNN_Trans(3,128,256)
dl3 = CNN_Trans(3,64,128)
final_cnn = CNN(3,64,1,tf_sigmoid,d_tf_sigmoid)

# graph
x = tf.placeholder(shape=[batch_size,64,64,3],dtype=tf.float32,name="input")
y = tf.placeholder(shape=[batch_size,64,64,1],dtype=tf.float32,name="output")

# encoder
elayer1 = el1.feedforward(x,padding='SAME')
elayer2_input = tf.nn.avg_pool(elayer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
elayer2 = el2.feedforward(elayer2_input,padding='SAME')
elayer3_input = tf.nn.avg_pool(elayer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
elayer3 = el3.feedforward(elayer3_input,padding='SAME')

elayer_mean =  e_mean.feedforward(elayer3)
elayer_var  =  e_var.feedforward(elayer3)
eps = tf.random_normal((elayer_var.shape), 0, 1, dtype=tf.float32)
z = elayer_mean + tf.sqrt(tf.exp(elayer_var)) * eps

# edcoder
dlayer1 = dl1.feedforward(z,stride=1,padding='SAME')
dlayer2 = dl2.feedforward(dlayer1,stride=2,padding='SAME')
dlayer3 = dl3.feedforward(dlayer2,stride=2,padding='SAME')
final_output = final_cnn.feedforward(dlayer3,padding='SAME')

# calculate the loss
cost = tf.reduce_mean(tf.square(final_output-y)) * 0.5
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    # start the training
    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)
        test_batch,test_label = shuffle(test_batch,test_label)

        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            current_batch_labek = train_label[batch_size_index:batch_size_index+batch_size]
            sess_result = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_batch_labek})
            print("Current Iter : ",iter ," current batch: ",batch_size_index, ' Current cost: ', sess_result[0],end='\r')
            train_cota = train_cota + sess_result[0]

        # For every print iteration save changes
        if iter % print_size==0:
            print("\n--------------")
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size)),end='\n')
            print("----------")

            # Get one example from training batch
            test_example = train_batch[:batch_size,:,:,:]
            test_example_gt = train_label[:batch_size,:,:,:]
            sess_results = sess.run([final_output],feed_dict={x:test_example})
            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]
            test_example_gt = test_example_gt[0,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example))
            plt.axis('off')
            plt.title('Original Image')
            plt.savefig('train_change_train/'+str(iter)+"a_Original_Image.png",bbox_inches='tight')
            plt.close('all')

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt),cmap='gray')
            plt.axis('off')
            plt.title('Original Image Mask')
            plt.savefig('train_change_train/'+str(iter)+"b_Original_Image_mask.png",bbox_inches='tight')
            plt.close('all')

            plt.figure()
            plt.imshow(test_example_gt*test_example)
            plt.axis('off')
            plt.title('Original Image Mask')
            plt.savefig('train_change_train/'+str(iter)+"c_Original_Image_overlay.png",bbox_inches='tight')
            plt.close('all')
  
            plt.figure()
            plt.imshow(np.squeeze(sess_results).astype(np.float32),cmap='gray')
            plt.axis('off')
            plt.title('Generated Mask')
            plt.savefig('train_change_train/'+str(iter)+"d_Generated_Mask.png",bbox_inches='tight')
            plt.close('all')

            plt.figure()
            plt.imshow(sess_results.astype(np.float32)*test_example)
            plt.axis('off')
            plt.title('Generated Mask')
            plt.savefig('train_change_train/'+str(iter)+"e_Generated_Mask_overlay.png",bbox_inches='tight')
            plt.close('all')

            # Get one Example from test batch
            test_example = test_batch[:batch_size,:,:,:]
            test_example_gt = test_label[:batch_size,:,:,:]
            sess_results = sess.run([final_output],feed_dict={x:test_example})
            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]
            test_example_gt = test_example_gt[0,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example))
            plt.axis('off')
            plt.title('Original Image')
            plt.savefig('train_change_test/'+str(iter)+"a_Original_Image.png",bbox_inches='tight')
            plt.close('all')

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt),cmap='gray')
            plt.axis('off')
            plt.title('Original Image Mask')
            plt.savefig('train_change_test/'+str(iter)+"b_Original_Image_mask.png",bbox_inches='tight')
            plt.close('all')

            plt.figure()
            plt.imshow(test_example_gt*test_example)
            plt.axis('off')
            plt.title('Original Image Mask')
            plt.savefig('train_change_test/'+str(iter)+"c_Original_Image_overlay.png",bbox_inches='tight')
            plt.close('all')
  
            plt.figure()
            plt.imshow(np.squeeze(sess_results).astype(np.float32),cmap='gray')
            plt.axis('off')
            plt.title('Generated Mask')
            plt.savefig('train_change_test/'+str(iter)+"d_Generated_Mask.png",bbox_inches='tight')
            plt.close('all')

            plt.figure()
            plt.imshow(sess_results.astype(np.float32)*test_example)
            plt.axis('off')
            plt.title('Generated Mask')
            plt.savefig('train_change_test/'+str(iter)+"e_Generated_Mask_overlay.png",bbox_inches='tight')
            plt.close('all')

        # sort the training error
        train_cot.append(train_cota/(len(train_batch)/(batch_size)))
        train_cota,train_acca = 0,0

    # Since all of the trainig is done create final test
    for batch_size_index in range(0,len(test_batch),batch_size):
        current_batch = test_batch[batch_size_index:batch_size_index+batch_size]
        current_batch_labek = test_label[batch_size_index:batch_size_index+batch_size]
        sess_result = sess.run(final_output,feed_dict={x:current_batch,y:current_batch_labek})

        for sess_index in range(len(sess_result)):
            sess_results = sess_result[sess_index]
            test_example = test_batch[sess_index,:,:,:]
            test_example_gt = test_label[sess_index,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example))
            plt.axis('off')
            plt.title('Original Image')
            plt.savefig('viz/'+str(batch_size_index)+str(sess_index)+"a_Original_Image.png",bbox_inches='tight')
            plt.close('all')

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt),cmap='gray')
            plt.axis('off')
            plt.title('Original Image Mask')
            plt.savefig('viz/'+str(batch_size_index)+str(sess_index)+"b_Original_Image_mask.png",bbox_inches='tight')
            plt.close('all')

            plt.figure()
            plt.imshow(test_example_gt*test_example)
            plt.axis('off')
            plt.title('Original Image Mask')
            plt.savefig('viz/'+str(batch_size_index)+str(sess_index)+"c_Original_Image_overlay.png",bbox_inches='tight')
            plt.close('all')
  
            plt.figure()
            plt.imshow(np.squeeze(sess_results).astype(np.float32),cmap='gray')
            plt.axis('off')
            plt.title('Generated Mask')
            plt.savefig('viz/'+str(batch_size_index)+str(sess_index)+"d_Generated_Mask.png",bbox_inches='tight')
            plt.close('all')

            plt.figure()
            plt.imshow(sess_results.astype(np.float32)*test_example)
            plt.axis('off')
            plt.title('Generated Mask')
            plt.savefig('viz/'+str(batch_size_index)+str(sess_index)+"e_Generated_Mask_overlay.png",bbox_inches='tight')
            plt.close('all')

    # Normalize the cost of the training
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))

    # plot the training and testing graph
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy/Cost Over Time")
    plt.savefig("viz/z_Case Train.png")
    plt.close('all')

# -- end code --