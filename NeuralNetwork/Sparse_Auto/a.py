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
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgba2rgb,rgb2gray

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(6278)
tf.set_random_seed(6728)
ia.seed(6278)

# ======= Activation Function  ==========
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
# ======= Activation Function  ==========

# ====== miscellaneous =====
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

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# ====== miscellaneous =====

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
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05,seed=2))
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
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05,seed=2))
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
        self.w = tf.Variable(tf.random_normal([input_dim,hidden_dim], stddev=0.05,seed=2))
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

class ICA_Layer():

    def __init__(self,inc):
        self.w_ica = tf.Variable(tf.random_normal([inc,inc],stddev=0.05,seed=2)) 
        # self.w_ica = tf.Variable(tf.eye(inc)*0.0001) 

    def feedforward(self,input):
        self.input = input
        self.ica_est = tf.matmul(input,self.w_ica)
        self.ica_est_act = tf_atan(self.ica_est)
        return self.ica_est_act

    def backprop(self):
        grad_part_2 = d_tf_atan(self.ica_est)
        grad_part_3 = self.input

        grad_pass = tf.matmul(grad_part_2,tf.transpose(self.w_ica))
        g_tf = tf.linalg.inv(tf.transpose(self.w_ica)) - (2/batch_size) * tf.matmul(tf.transpose(self.input),self.ica_est_act)

        update_w = []
        update_w.append(tf.assign(self.w_ica,self.w_ica+0.2*g_tf))

        return grad_pass,update_w  

# ================= LAYER CLASSES =================

# data
PathDicom = "../../Dataset/PennFudanPed/PNGImages/"
image_list = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if  ".png" in filename.lower() :  # check whether the file's DICOM \PedMasks
            image_list.append(os.path.join(dirName,filename))

mask_list = []  # create an empty list
PathDicom = "../../Dataset/PennFudanPed/PedMasks/"
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if  ".png" in filename.lower() :  # check whether the file's DICOM \PedMasks
            mask_list.append(os.path.join(dirName,filename))

image_resize_px = 128
train_images = np.zeros(shape=(170,image_resize_px,image_resize_px,3))
train_labels = np.zeros(shape=(170,image_resize_px,image_resize_px,1))

for file_index in range(len(image_list)):
    train_images[file_index,:,:]   = imresize(imread(image_list[file_index],mode='RGB'),(image_resize_px,image_resize_px))
    train_labels[file_index,:,:]   = np.expand_dims(imresize(rgb2gray(imread(mask_list[file_index],mode='RGB')),(image_resize_px,image_resize_px)),3) * 255.0

print(len(train_images))
print(len(train_labels))

for xx in range(len(train_images)):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(train_images[xx].astype(int))
    ax2.imshow(np.squeeze(train_labels[xx]).astype(int))
    plt.show()

# # Normalize data from 0 to 1 per each channel
# train_batch[:,:,:,0]  = (train_batch[:,:,:,0] - train_batch[:,:,:,0].min(axis=0)) / (train_batch[:,:,:,0].max(axis=0) - train_batch[:,:,:,0].min(axis=0))
# train_batch[:,:,:,1]  = (train_batch[:,:,:,1] - train_batch[:,:,:,1].min(axis=0)) / (train_batch[:,:,:,1].max(axis=0) - train_batch[:,:,:,1].min(axis=0))
# train_batch[:,:,:,2]  = (train_batch[:,:,:,2] - train_batch[:,:,:,2].min(axis=0)) / (train_batch[:,:,:,2].max(axis=0) - train_batch[:,:,:,2].min(axis=0))

# test_batch[:,:,:,0]  = (test_batch[:,:,:,0] - test_batch[:,:,:,0].min(axis=0)) / (test_batch[:,:,:,0].max(axis=0) - test_batch[:,:,:,0].min(axis=0))
# test_batch[:,:,:,1]  = (test_batch[:,:,:,1] - test_batch[:,:,:,1].min(axis=0)) / (test_batch[:,:,:,1].max(axis=0) - test_batch[:,:,:,1].min(axis=0))
# test_batch[:,:,:,2]  = (test_batch[:,:,:,2] - test_batch[:,:,:,2].min(axis=0)) / (test_batch[:,:,:,2].max(axis=0) - test_batch[:,:,:,2].min(axis=0))

# # print out the data shape
# print(train_batch.shape)
# print(train_label.shape)
# print(test_batch.shape)
# print(test_label.shape)

sys.exit()

# class
el1 = CNN(3,1,16)
el2 = CNN(3,16,32)
el3 = CNN(3,32,64)
el4 = FNN(7*7*64,256,act=tf_atan,d_act=d_tf_atan)

ica_l0 = FNN(256,3,act=tf_atan,d_act=d_tf_atan)
ica_l1 = ICA_Layer(3)

dl0 = FNN(256,7*7*128,act=tf_iden,d_act=d_tf_iden)
dl1 = CNN_Trans(3,64,128)
dl2 = CNN_Trans(3,32,64)
dl3 = CNN_Trans(3,16,32)

fl1 = CNN(3,32,32)
fl2 = CNN(3,16,1,act=tf_sigmoid,d_act=d_tf_sigmoid)

# hyper
num_epoch = 101
learning_rate = 0.0001
batch_size = 200
print_size = num_epoch

beta1,beta2,adam_e = 0.9,0.9,1e-8

# graph
x = tf.placeholder(shape=[batch_size,28,28,1],dtype=tf.float32)

elayer1 = el1.feedforward(x)
elayer2_input = tf.nn.avg_pool(elayer1,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID')
elayer2 = el2.feedforward(elayer2_input)

elayer3_input = tf.nn.avg_pool(elayer2,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID')
elayer3 = el3.feedforward(elayer3_input)

elayer4_input = tf.reshape(elayer3,[batch_size,-1])
elayer4 = el4.feedforward(elayer4_input)

ica_layer0 = ica_l0.feedforward(elayer4)

ica_layer1_temp = []
for _ in range(int(num_epoch*3)):
    ica_layer1 = ica_l1.feedforward(ica_layer0)
    grad_ica1,grad_ica_up1 = ica_l1.backprop()
    ica_layer1_temp.append(grad_ica_up1)

dlayer0 = dl0.feedforward(elayer4)
dlayer1_input = tf.reshape(dlayer0,[batch_size,7,7,128])

dlayer1 = dl1.feedforward(dlayer1_input,stride=2)

dlayer2 = dl2.feedforward(dlayer1,stride=2)
flayer1 = fl1.feedforward(dlayer2,stride=2)

dlayer3 = dl3.feedforward(flayer1,stride=2)
flayer2 = fl2.feedforward(dlayer3)

cost0 = tf.reduce_mean(tf.square(flayer2-x)*0.5)
total_cost = cost0

# auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_cost)

grad_fl2,grad_fl2_up = fl2.backprop(flayer2-x)
grad_dl3,grad_dl3_up = dl3.backprop(grad_fl2,stride=2)

grad_fl1,grad_fl1_up = fl1.backprop(grad_dl3,stride=2)
grad_dl2,grad_dl2_up = dl2.backprop(grad_fl1,stride=2)

grad_dl1,grad_dl1_up = dl1.backprop(grad_dl2,stride=2)

grad_d0_input = tf.reshape(grad_dl1,[batch_size,-1])
grad_dl0,grad_dl0_up = dl0.backprop(grad_d0_input)

# grad_ica1,grad_ica_up1 = ica_l1.backprop()
grad_ica0,grad_ica_up0 = ica_l0.backprop(grad_ica1)

grad_el4,grad_el4_up = el4.backprop(grad_dl0+grad_ica0)
grad_el3_input = tf.reshape(grad_el4,[batch_size,7,7,64])

grad_el3,grad_el3_up = el3.backprop(grad_el3_input)
grad_el2_input = tf_repeat(grad_el3,[1,2,2,1])

grad_el2,grad_el2_up = el2.backprop(grad_el2_input)
grad_el1_input = tf_repeat(grad_el2,[1,2,2,1])

grad_el1,grad_el1_up = el1.backprop(grad_el1_input)

grad_update0 = grad_fl2_up + grad_dl3_up + \
              grad_fl1_up + grad_dl2_up + \
              grad_dl1_up + grad_dl0_up + \
              grad_el4_up + grad_el3_up + \
              grad_el2_up + grad_el1_up 

grad_update1 = grad_fl2_up + grad_dl3_up + \
              grad_fl1_up + grad_dl2_up + \
              ica_layer1_temp + grad_ica_up0 + \
              grad_dl1_up + grad_dl0_up + \
              grad_el4_up + grad_el3_up + \
              grad_el2_up + grad_el1_up 

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    # start the training
    for iter in range(num_epoch):

        if iter > num_epoch//3:
            sess_aim = [total_cost,grad_update1]
        else:
            sess_aim = [total_cost,grad_update0]

        # train for batch
        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            sess_result = sess.run(sess_aim,feed_dict={x:current_batch})
            print("Current Iter : ",iter ," current batch: ",batch_size_index, ' Current cost: ', sess_result[0],end='\r')
            train_cota = train_cota + sess_result[0]

        # if it is print size print the cost and Sample Image
        if iter % print_size==0:
            print("\n--------------")   
            print('Current Iter: ',iter,' Accumulated Train cost : ', train_cota/(len(train_batch)/(batch_size)),end='\n')
            print("--------------")

            # get one image from train batch and show results
            test_example = train_batch[:batch_size,:,:,:]
            sess_results = sess.run([flayer2],feed_dict={x:test_example})
            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]

            plt.figure(1, figsize=(18,9))
            plt.suptitle('Original Image (left) Generated Image (right) Iter: ' + str(iter),fontsize=20)
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(np.squeeze(test_example),cmap='gray')
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(np.squeeze(sess_results).astype(np.float32),cmap='gray')
            plt.savefig('train_change/'+str(iter)+"_train_results.png",bbox_inches='tight')
            plt.close('all')

            # get one image from test batch and show results
            test_example = test_batch[:batch_size,:,:,:]
            sess_results = sess.run([flayer2],feed_dict={x:test_example})
            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]

            plt.figure(1, figsize=(18,9))
            plt.suptitle('Original Image (left) Generated Image (right) Iter: ' + str(iter),fontsize=20)
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(np.squeeze(test_example),cmap='gray')
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(np.squeeze(sess_results).astype(np.float32),cmap='gray')
            plt.savefig('test_change/'+str(iter)+"_test_results.png",bbox_inches='tight')
            plt.close('all')

        train_cot.append(train_cota/(len(train_batch)/(batch_size)))
        train_cota,train_acca = 0,0

    # Normalize the cost of the training
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))

    # plot the training and testing graph
    plt.figure()
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig("viz/Case Train.png")
    plt.close('all')

    # train image final show
    final_train = sess.run([flayer2,ica_layer1],feed_dict={x:train_batch[:batch_size,:,:,:]}) 
    final_train_ica = final_train[1]
    final_train = final_train[0]
    for current_image_index in range(batch_size//100):
        plt.figure(1, figsize=(18,9))
        plt.suptitle('Original Image (left) Generated Image (right) image num : ' + str(iter))
        plt.subplot(121)
        plt.axis('off')
        plt.imshow(np.squeeze(train_batch[current_image_index]))
        plt.subplot(122)
        plt.axis('off')
        plt.imshow(np.squeeze(final_train[current_image_index]).astype(np.float32))
        plt.savefig('final_train/'+str(current_image_index)+"_train_results.png",bbox_inches='tight',fontsize=20)
        plt.close('all')
        
    # test image final show
    final_test = sess.run([flayer2,ica_layer1],feed_dict={x:test_batch[:batch_size,:,:,:]}) 
    final_test_ica = final_test[1]
    final_test = final_test[0]
    for current_image_index in range(batch_size//100):
        plt.figure(1, figsize=(18,9))
        plt.suptitle('Original Image (left) Generated Image (right) image num : ' + str(iter))
        plt.subplot(121)
        plt.axis('off')
        plt.imshow(np.squeeze(test_batch[current_image_index]))
        plt.subplot(122)
        plt.axis('off')
        plt.imshow(np.squeeze(final_test[current_image_index]).astype(np.float32))
        plt.savefig('final_test/'+str(current_image_index)+"_test_results.png",bbox_inches='tight',fontsize=20)
        plt.close('all')

    # plot 3D 
    color_dict = {
        0:'red',
        1:'blue',
        2:'green',
        3:'yellow',
        4:'purple',
        5:'grey',
        6:'black',
        7:'white',
        8:'pink',
        9:'skyblue',
    }

    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    color_mapping = [color_dict[x] for x in np.argmax(train_label[:batch_size],1) ]
    ax.scatter(final_train_ica[:,0], final_train_ica[:,1],final_train_ica[:,2],c=color_mapping,label=str(color_dict))

    color_mapping = [color_dict[x] for x in np.argmax(test_label[:batch_size],1) ]
    ax.scatter(final_test_ica[:,0], final_test_ica[:,1],final_test_ica[:,2],c=color_mapping)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    ax.legend()
    ax.grid(True)
    plt.show()
# -- end code --