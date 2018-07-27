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
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05,seed=2,dtype=tf.float64))
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
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05,seed=2,dtype=tf.float64))
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
        self.w = tf.Variable(tf.random_normal([input_dim,hidden_dim], stddev=0.05,seed=2,dtype=tf.float64))
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

class Sparse_Filter_Layer():
    
    def __init__(self,outc,changec):
        self.w = tf.Variable(tf.truncated_normal([outc,changec],stddev=0.5,seed=2,dtype=tf.float64))
        self.epsilon = 1e-20

    def getw(self): return self.w

    def soft_abs(self,value):
        return tf.sqrt(value ** 2 + self.epsilon)

    def feedforward(self,input):
        self.sparse_layer  = tf.matmul(input,self.w)
        # second = tf.nn.elu(self.sparse_layer)
        second = self.soft_abs(self.sparse_layer )
        third  = tf.divide(second,tf.sqrt(tf.reduce_sum(second**2,axis=0)+self.epsilon))
        four = tf.divide(third,tf.sqrt(tf.reduce_sum(third**2,axis=1)[:,tf.newaxis] +self.epsilon))
        self.cost_update = tf.reduce_mean(four)
        return self.sparse_layer ,self.cost_update

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

image_resize_px = 96    
train_images = np.zeros(shape=(170,image_resize_px,image_resize_px,3))
train_labels = np.zeros(shape=(170,image_resize_px,image_resize_px,1))

for file_index in range(len(image_list)):
    train_images[file_index,:,:]   = imresize(imread(image_list[file_index],mode='RGB'),(image_resize_px,image_resize_px))
    train_labels[file_index,:,:]   = np.expand_dims(imresize(rgb2gray(imread(mask_list[file_index],mode='RGB')),(image_resize_px,image_resize_px)),3) 

train_labels = (train_labels>25.0) * 255.0
train_images = train_images/255.0
train_labels = train_labels/255.0

train_batch = train_images[:40]
train_label = train_labels[:40]
test_batch = train_images[40:]
test_label = train_labels[40:]

# print out the data shape
print(train_batch.shape)
print(train_batch.max())
print(train_batch.min())
print(train_label.shape)
print(train_label.max())
print(train_label.min())

print(test_batch.shape)
print(test_batch.max())
print(test_batch.min())
print(test_label.shape)
print(test_label.max())
print(test_label.min())

# class
el1 = CNN(3,3,11)
el2 = CNN(3,11,11)
el3 = CNN(3,11,11)
el4 = CNN(3,11,11)

reduce_dim = 25
sparse_layer = Sparse_Filter_Layer(6*6*11,1*1*reduce_dim)

dl0 = CNN_Trans(5,9,1,act=tf_sigmoid)
dl1 = CNN_Trans(3,9,9)
fl1 = CNN(1,9,9,act=tf_sigmoid)

dl2 = CNN_Trans(5,9,20)
fl2 = CNN(1,9,9,act=tf_sigmoid)

dl3 = CNN_Trans(3,9,20)
fl3 = CNN(1,9,9,act=tf_sigmoid)

dl4 = CNN_Trans(3,4,20)
fl4 = CNN(1,4,1,act=tf_sigmoid)

# hyper
num_epoch = 1201
learning_rate = 0.0008
batch_size = 10
print_size = 100

# graph
x = tf.placeholder(shape=[batch_size,image_resize_px,image_resize_px,3],dtype=tf.float64)
y = tf.placeholder(shape=[batch_size,image_resize_px,image_resize_px,1],dtype=tf.float64)

elayer1 = el1.feedforward(x)

elayer2_input = tf.nn.max_pool(elayer1,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID')
elayer2 = el2.feedforward(elayer2_input)

elayer3_input = tf.nn.max_pool(elayer2,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID')
elayer3 = el3.feedforward(elayer3_input)

elayer4_input = tf.nn.max_pool(elayer3,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID')
elayer4 = el4.feedforward(elayer4_input)

sparse_input = tf.nn.max_pool(elayer4,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID')
sparse_layer_input = tf.reshape(sparse_input,[batch_size,-1])

sparse_layer_value0,sparse_cost0 = sparse_layer.feedforward(sparse_layer_input)
sparse_layer_value1,sparse_cost1 = sparse_layer.feedforward(sparse_layer_input)
sparse_layer_value2,sparse_cost2 = sparse_layer.feedforward(sparse_layer_input)
sparse_layer_value3,sparse_cost3 = sparse_layer.feedforward(sparse_layer_input)
sparse_layer_value4,sparse_cost4 = sparse_layer.feedforward(sparse_layer_input)
sparse_layer_value5,sparse_cost5 = sparse_layer.feedforward(sparse_layer_input)
sparse_layer_value = sparse_layer_value0 + sparse_layer_value1 + sparse_layer_value2 +sparse_layer_value3+sparse_layer_value4+sparse_layer_value5

dlayer0_input = tf.reshape(sparse_layer_value,[batch_size,5,5,1])
dlayer0_input = tf.image.resize_images(dlayer0_input, [6, 6],method=tf.image.ResizeMethod.BILINEAR,align_corners=False)
dlayer0_input2 = tf.cast(dlayer0_input,dtype=tf.float64)
dlayer0 = dl0.feedforward(dlayer0_input2,stride=1) 

dlayer01 = tf.image.resize_images(dlayer0, [12, 12],method=tf.image.ResizeMethod.BICUBIC,align_corners=False)
dlayer01 = tf.cast(dlayer01,dtype=tf.float64)
dlayer1 = dl1.feedforward(dlayer01) 
flayer1 = fl1.feedforward(dlayer1)

flayer11 = tf.image.resize_images(flayer1, [24, 24],method=tf.image.ResizeMethod.BILINEAR,align_corners=False)
flayer11 = tf.cast(flayer11,dtype=tf.float64)
dlayer2 = dl2.feedforward(tf.concat([flayer11,elayer3],3),stride=1) 
flayer2 = fl2.feedforward(dlayer2)

flayer21 = tf.image.resize_images(flayer2, [48, 48],method=tf.image.ResizeMethod.BICUBIC,align_corners=False)
flayer21 = tf.cast(flayer21,dtype=tf.float64)
dlayer3 = dl3.feedforward(tf.concat([flayer21,elayer2],3))
flayer3 = fl3.feedforward(dlayer3)

flayer31 = tf.image.resize_images(flayer3, [96, 96],method=tf.image.ResizeMethod.BILINEAR,align_corners=False)
flayer31 = tf.cast(flayer31,dtype=tf.float64)
dlayer4 = dl4.feedforward(tf.concat([flayer31,elayer1],3),stride=1)
flayer4 = fl4.feedforward(dlayer4)

cost0 = tf.reduce_mean(tf.square(flayer4-y))
cost1 = tf.reduce_mean([sparse_cost0 ,sparse_cost1 ,sparse_cost2,sparse_cost3,sparse_cost4,sparse_cost5])

total_cost = cost0 + cost1
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_cost)

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

        # train_batch,train_label = shuffle(train_batch,train_label)
        test_batch,test_label = shuffle(test_batch,test_label)

        # train for batch
        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size]
            sess_result = sess.run([total_cost,auto_train],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter : ",iter ,' Current cost: ', sess_result[0],end='\r')
            train_cota = train_cota + sess_result[0]

        # if it is print size print the cost and Sample Image
        if iter % print_size==0:
            print("\n--------------")   
            print('Current Iter: ',iter,' Accumulated Train cost : ', train_cota/(len(train_batch)/(batch_size)),end='\n')
            print("--------------")

            # get one image from train batch and show results
            sess_results = sess.run(flayer4,feed_dict={x:train_batch[:batch_size]})
            test_change_image = train_batch[0,:,:,:]
            test_change_gt = train_label[0,:,:,:]
            test_change_predict = sess_results[0,:,:,:]

            f, axarr = plt.subplots(2, 3,figsize=(27,18))
            plt.suptitle('Original Image (left) Generated Image (right) Iter: ' + str(iter),fontsize=20)
            axarr[0, 0].axis('off')
            axarr[0, 0].imshow(np.squeeze(test_change_image),cmap='gray')

            axarr[0, 1].axis('off')
            axarr[0, 1].imshow(np.squeeze(test_change_gt),cmap='gray')

            axarr[0, 2].axis('off')
            axarr[0, 2].imshow(np.squeeze(test_change_predict),cmap='gray')

            axarr[1, 0].axis('off')
            axarr[1, 0].imshow(np.squeeze(test_change_image),cmap='gray')

            axarr[1, 1].axis('off')
            axarr[1, 1].imshow(test_change_gt*np.squeeze(test_change_image),cmap='gray')

            axarr[1, 2].axis('off')
            axarr[1, 2].imshow(test_change_predict*np.squeeze(test_change_image),cmap='gray')

            plt.savefig('train_change/'+str(iter)+"_train_results.png",bbox_inches='tight')
            plt.close('all')

            # get one image from test batch and show results
            sess_results = sess.run(flayer4,feed_dict={x:test_batch[:batch_size]})
            test_change_image = test_batch[:batch_size][0,:,:,:]
            test_change_gt = test_label[0,:,:,:]
            test_change_predict = sess_results[0,:,:,:]

            f, axarr = plt.subplots(2, 3,figsize=(27,18))
            plt.suptitle('Original Image (left) Generated Image (right) Iter: ' + str(iter),fontsize=20)
            axarr[0, 0].axis('off')
            axarr[0, 0].imshow(np.squeeze(test_change_image),cmap='gray')

            axarr[0, 1].axis('off')
            axarr[0, 1].imshow(np.squeeze(test_change_gt),cmap='gray')

            axarr[0, 2].axis('off')
            axarr[0, 2].imshow(np.squeeze(test_change_predict),cmap='gray')

            axarr[1, 0].axis('off')
            axarr[1, 0].imshow(np.squeeze(test_change_image),cmap='gray')

            axarr[1, 1].axis('off')
            axarr[1, 1].imshow(test_change_gt*np.squeeze(test_change_image),cmap='gray')

            axarr[1, 2].axis('off')
            axarr[1, 2].imshow(test_change_predict*np.squeeze(test_change_image),cmap='gray')

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

    # final all train images
    for batch_size_index in range(0,len(train_batch),batch_size):
        current_batch = train_batch[batch_size_index:batch_size_index+batch_size]    
        current_batch_label = train_label[batch_size_index:batch_size_index+batch_size]
        sess_results = sess.run(flayer4,feed_dict={x:current_batch})
        for xx in range(len(sess_results)):
            f, axarr = plt.subplots(2, 3,figsize=(27,18))

            test_change_predict = sess_results[xx]

            plt.suptitle('Final Train Images : ' + str(xx) ,fontsize=20)
            axarr[0, 0].axis('off')
            axarr[0, 0].imshow(np.squeeze(current_batch[xx]),cmap='gray')

            axarr[0, 1].axis('off')
            axarr[0, 1].imshow(np.squeeze(current_batch_label[xx]),cmap='gray')

            axarr[0, 2].axis('off')
            axarr[0, 2].imshow(np.squeeze(test_change_predict),cmap='gray')

            axarr[1, 0].axis('off')
            axarr[1, 0].imshow(np.squeeze(current_batch[xx]),cmap='gray')

            axarr[1, 1].axis('off')
            axarr[1, 1].imshow(current_batch_label[xx]*np.squeeze(current_batch[xx]),cmap='gray')

            axarr[1, 2].axis('off')
            axarr[1, 2].imshow(test_change_predict*np.squeeze(current_batch[xx]),cmap='gray')

            plt.savefig('final_train/'+str(batch_size_index)+"_"+str(xx)+"_train_results.png",bbox_inches='tight')
            plt.close('all')


    for batch_size_index in range(0,len(test_batch),batch_size):
        current_batch = test_batch[batch_size_index:batch_size_index+batch_size]    
        current_batch_label = test_label[batch_size_index:batch_size_index+batch_size]
        sess_results = sess.run(flayer4,feed_dict={x:current_batch})
        for xx in range(len(sess_results)):
            f, axarr = plt.subplots(2, 3,figsize=(27,18))
        
            test_change_predict = sess_results[xx]

            plt.suptitle('Final Test Images : ' + str(xx) ,fontsize=20)
            axarr[0, 0].axis('off')
            axarr[0, 0].imshow(np.squeeze(current_batch[xx]),cmap='gray')

            axarr[0, 1].axis('off')
            axarr[0, 1].imshow(np.squeeze(current_batch_label[xx]),cmap='gray')

            axarr[0, 2].axis('off')
            axarr[0, 2].imshow(np.squeeze(test_change_predict),cmap='gray')

            axarr[1, 0].axis('off')
            axarr[1, 0].imshow(np.squeeze(current_batch[xx]),cmap='gray')

            axarr[1, 1].axis('off')
            axarr[1, 1].imshow(current_batch_label[xx]*np.squeeze(current_batch[xx]),cmap='gray')

            axarr[1, 2].axis('off')
            axarr[1, 2].imshow(test_change_predict*np.squeeze(current_batch[xx]),cmap='gray')

            plt.savefig('final_test/'+str(batch_size_index)+"_"+str(xx)+"_test_results.png",bbox_inches='tight')
            plt.close('all')


# -- end code --