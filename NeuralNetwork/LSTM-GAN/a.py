import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread,imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from imgaug import augmenters as iaa
import nibabel as nib
import imgaug as ia

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
def softabs(x): return tf.sqrt(x ** 2 + 1e-20)
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

# ================= LAYER CLASSES =================
class CNN():
    
    def __init__(self,k,inc,out,act=tf_elu,d_act=d_tf_elu):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05,seed=2,dtype=tf.float64))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act

    def getw(self): return [self.w]

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

class CNN_3D():
    
    def __init__(self,filter_depth,filter_height,filter_width,in_channels,out_channels,act=tf_elu,d_act=d_tf_elu):
        self.w = tf.Variable(tf.random_normal([filter_depth,filter_height,filter_width,in_channels,out_channels],stddev=0.05,seed=2,dtype=tf.float64))
        self.act,self.d_act = act,d_act
    def getw(self): return [self.w]
    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv3d(input,self.w,strides=[1,1,1,1,1],padding=padding)
        self.layerA = self.act(self.layer)
        return self.layerA 


class CNN_Trans():
    
    def __init__(self,k,inc,out,act=tf_elu,d_act=d_tf_elu):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05,seed=2,dtype=tf.float64))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act

    def getw(self): return [self.w]

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

    def getw(self): return [self.w]

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
        self.w = tf.Variable(tf.random_normal([outc,changec],stddev=1.0,seed=2,dtype=tf.float64))
        self.epsilon = 1e-20

    def getw(self): return self.w

    def soft_abs(self,value):
        return tf.sqrt(value ** 2 + self.epsilon)

    def feedforward(self,input):
        self.sparse_layer  = tf.matmul(input,self.w)
        second = self.soft_abs(self.sparse_layer )
        third  = tf.divide(second,tf.sqrt(tf.reduce_sum(second**2,axis=0)+self.epsilon))
        four = tf.divide(third,tf.sqrt(tf.reduce_sum(third**2,axis=1)[:,tf.newaxis] +self.epsilon))
        self.cost_update = tf.reduce_mean(four)
        return self.sparse_layer ,self.cost_update
# ================= LAYER CLASSES =================

# data
PathDicom = "../../Dataset/Neurofeedback_Skull_stripped/NFBS_Dataset/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".nii.gz" in filename.lower() and not 'brain' in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
all_brain_data = np.zeros((20,192,256,256))
for current_brain in range(len(all_brain_data)):
    all_brain_data[current_brain] = nib.load(lstFilesDCM[current_brain]).get_fdata().T 
all_brain_data = all_brain_data/all_brain_data.max(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]

PathDicom = "../../Dataset/Neurofeedback_Skull_stripped/NFBS_Dataset/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".nii.gz" in filename.lower() and  'brainmask' in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
all_mask_data = np.zeros((20,192,256,256))
for current_brain in range(len(all_mask_data)):
    all_mask_data[current_brain] = nib.load(lstFilesDCM[current_brain]).get_fdata().T 
all_mask_data = all_mask_data/all_mask_data.max(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]

split_number = 18
train_batch = all_brain_data[:split_number]
train_label = all_mask_data[:split_number]
test_batch = all_brain_data[split_number:]
test_label =all_mask_data[split_number:]

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
l0 = CNN_3D(3,3,3,1,3)
l1 = CNN_3D(3,3,3,3,6)
l2 = CNN_3D(3,3,3,6,6)
l3 = CNN_3D(3,3,3,6,3)
l4 = CNN_3D(3,3,3,3,1)

# hyper
num_epoch = 801
learning_rate = 0.0005
batch_size = 2
print_size = 10

# graph
x = tf.placeholder(shape=(batch_size,192,256,256,1),dtype=tf.float64)
y = tf.placeholder(shape=(batch_size,192,256,256,1),dtype=tf.float64)

layer0 = l0.feedforward(x)
layer1 = l1.feedforward(layer0)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)
layer4 = l4.feedforward(layer3)

cost1 = tf.reduce_mean(tf.square(layer4-y))
cost2 = -tf.reduce_mean( y * tf.log(layer4 + 1e-20) + (1.0-y)*tf.log(1-layer4 + 1e-20) )
total_cost = cost1 + cost2
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_cost)

sys.exit()

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

        which_opt = auto_train
        which_cost = total_cost

        train_batch,train_label = shuffle(train_batch,train_label)
        test_batch,test_label = shuffle(test_batch,test_label)

        # train for batch
        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size]
            sess_result = sess.run([which_cost,which_opt],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter : ",iter ,' Current cost: ', sess_result[0],end='\r')
            train_cota = train_cota + sess_result[0]

        # if it is print size print the cost and Sample Image
        if iter % print_size==0:
            print("\n--------------")   
            print('Current Iter: ',iter,' Accumulated Train cost : ', train_cota/(len(train_batch)/(batch_size)),end='\n')
            print("--------------")

        if iter % print_size==0 and iter > num_to_change:

            # get one image from train batch and show results
            sess_results = sess.run(flayer5,feed_dict={x:train_batch[:batch_size]})
            test_change_image = train_batch[0,:,:,:]
            test_change_gt = train_label[0,:,:,:]
            test_change_predict = sess_results[0,:,:,:]

            f, axarr = plt.subplots(2, 3,figsize=(30,20))
            plt.suptitle('Current Iter' + str(iter),fontsize=20)
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
            sess_results = sess.run(flayer5,feed_dict={x:test_batch[:batch_size]})
            test_change_image = test_batch[:batch_size][0,:,:,:]
            test_change_gt = test_label[0,:,:,:]
            test_change_predict = sess_results[0,:,:,:]

            f, axarr = plt.subplots(2, 3,figsize=(30,20))
            plt.suptitle('Current Iter' + str(iter),fontsize=20)
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
        sess_results = sess.run(flayer5,feed_dict={x:current_batch})
        for xx in range(len(sess_results)):
            f, axarr = plt.subplots(2, 3,figsize=(27,18))

            # test_change_predict = (sess_results[xx]-sess_results[xx].min())/(sess_results[xx].max()-sess_results[xx].min())
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

    # final all test images
    for batch_size_index in range(0,len(test_batch),batch_size):
        current_batch = test_batch[batch_size_index:batch_size_index+batch_size]    
        current_batch_label = test_label[batch_size_index:batch_size_index+batch_size]
        sess_results = sess.run(flayer5,feed_dict={x:current_batch})
        for xx in range(len(sess_results)):
            f, axarr = plt.subplots(2, 3,figsize=(27,18))
        
            # test_change_predict = (sess_results[xx]-sess_results[xx].min())/(sess_results[xx].max()-sess_results[xx].min())
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

    # final all mall images
    for batch_size_index in range(0,len(mall_data),batch_size):
        current_batch = mall_data[batch_size_index:batch_size_index+batch_size]    
        sess_results = sess.run(flayer5,feed_dict={x:current_batch})
        for xx in range(len(sess_results)):
            test_change_predict = sess_results[xx]
            plt.figure(figsize=(8, 8))    
            plt.imshow(np.squeeze(current_batch[xx]),cmap='gray')
            plt.imshow(np.squeeze(test_change_predict), cmap='hot', alpha=0.5)
            plt.axis('off')
            plt.savefig('mall_frame/'+str(batch_size_index)+"_"+str(xx)+".png",bbox_inches='tight')
            plt.close('all')




# -- end code --