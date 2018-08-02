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
from scipy.ndimage import zoom

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
        self.b = tf.Variable(tf.random_normal([out_channels],stddev=0.05,seed=2,dtype=tf.float64))
        self.act,self.d_act = act,d_act
    def getw(self): return [self.w]
    def feedforward(self,input,stride=1,padding='SAME',res=True):
        self.input  = input
        self.layer  = tf.nn.conv3d(input,self.w,strides=[1,1,1,1,1],padding=padding) + self.b
        self.layerA = self.act(self.layer)
        if res:
            return self.layerA + self.input
        else:
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

# read the data
all_brain_data= np.load('all_brain_data.npy')
all_mask_data = np.load('all_mask_data.npy')

all_brain_data = (all_brain_data-all_brain_data.min(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis] ) / \
                (all_brain_data.max(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis] - all_brain_data.min(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis] +1e-10 ) 

all_mask_data = (all_mask_data-all_mask_data.min(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis] ) / \
                (all_mask_data.max(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis] - all_mask_data.min(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis] +1e-10 ) 

split_number = 18
train_batch = all_brain_data[:split_number]
train_label = all_mask_data[:split_number]
test_batch = all_brain_data[split_number:]
test_label =all_mask_data[split_number:]

# print out the data shape
print('-----------------------')
print(train_batch.shape)
print(train_batch.max(axis=(1,2,3)).max())
print(train_batch.min(axis=(1,2,3)).max())
print(train_label.shape)
print(train_label.max(axis=(1,2,3)).max())
print(train_label.min(axis=(1,2,3)).max())

print(test_batch.shape)
print(test_batch.max(axis=(1,2,3)).max())
print(test_batch.min(axis=(1,2,3)).max())
print(test_label.shape)
print(test_label.max(axis=(1,2,3)).max())
print(test_label.min(axis=(1,2,3)).max())
print('-----------------------')

# class
g0 = CNN_3D(3,3,3,1,6)
g1 = CNN_3D(3,3,3,6,6)
g2 = CNN_3D(3,1,1,6,6)
g3 = CNN_3D(3,3,3,6,6)
g4 = CNN_3D(3,3,3,6,1,act=tf_sigmoid)

d0 = CNN_3D(3,3,3,1,6)
d1 = CNN_3D(3,3,3,6,6)
d2 = CNN_3D(3,1,1,6,6)
d3 = CNN_3D(3,3,3,6,6)
d4 = CNN_3D(3,3,3,6,1,act=tf_sigmoid)

# hyper
num_epoch = 20
learning_rate = 0.0009
batch_size = 2
print_size = 1
divide_size = 3

# graph
x = tf.placeholder(shape=(batch_size,divide_size,32,32,1),dtype=tf.float64)
y = tf.placeholder(shape=(batch_size,divide_size,32,32,1),dtype=tf.float64)

layer0_g = g0.feedforward(x,res=False)
layer1_g = g1.feedforward(layer0_g)
layer2_g = g2.feedforward(layer1_g)
layer3_g = g3.feedforward(layer2_g)
layer4_g = g4.feedforward(layer3_g,res=False)

true_d_1 = d0.feedforward(y,res=False)
true_d_2 = d1.feedforward(true_d_1)
true_d_3 = d2.feedforward(true_d_2)
true_d_4 = d3.feedforward(true_d_3)
true_d_f = d4.feedforward(true_d_4,res=False)

fake_d_1 = d0.feedforward(layer4_g,res=False)
fake_d_2 = d1.feedforward(fake_d_1)
fake_d_3 = d2.feedforward(fake_d_2)
fake_d_4 = d3.feedforward(fake_d_3)
fake_d_f = d4.feedforward(fake_d_4,res=False)
d_weights = d0.getw() + d1.getw() + d2.getw() + d3.getw() + d4.getw() 
g_weights = g0.getw() + g1.getw() + g2.getw() + g3.getw() + g4.getw() 

# ----- losses
D_loss = -tf.reduce_mean(tf.log(true_d_f+1e-10) + tf.log(1.0 - fake_d_f+1e-10))
G_loss = -tf.reduce_mean(tf.log(fake_d_f+1e-10)) 

# --- training
auto_d_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss,var_list= d_weights)
auto_g_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss,var_list= g_weights)

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

        # train for batch
        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size]
            for divide_batch_index in range(0,192,divide_size):
                current_batch_divide = current_batch[:,divide_batch_index:divide_batch_index+divide_size,:,:,:]
                current_batch_label_divide = current_batch_label[:,divide_batch_index:divide_batch_index+divide_size,:,:,:]

                sess_result = sess.run([D_loss,auto_d_train],feed_dict={x:current_batch_divide,y:current_batch_label_divide})
                print("Current Iter : ",iter,' current batch: ',batch_size_index ,' current divide index : ',divide_batch_index ,' Current cost: ', sess_result[0],end='\r')
                train_cota = train_cota + sess_result[0]

                sess_result = sess.run([G_loss,auto_g_train],feed_dict={x:current_batch_divide,y:current_batch_label_divide})
                print("Current Iter : ",iter,' current batch: ',batch_size_index ,' current divide index : ',divide_batch_index ,' Current cost: ', sess_result[0],end='\r')
                train_cota = train_cota + sess_result[0]

        # test for batch
        for batch_size_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[batch_size_index:batch_size_index+batch_size]
            current_batch_label = test_label[batch_size_index:batch_size_index+batch_size]
            for divide_batch_index in range(0,192,divide_size):
                current_batch_divide = current_batch[:,divide_batch_index:divide_batch_index+divide_size,:,:,:]
                current_batch_label_divide = current_batch_label[:,divide_batch_index:divide_batch_index+divide_size,:,:,:]

                sess_result = sess.run([D_loss],feed_dict={x:current_batch_divide,y:current_batch_label_divide})
                print("Current Iter : ",iter,' current batch: ',batch_size_index ,' current divide index : ',divide_batch_index ,' Current cost: ', sess_result[0],end='\r')
                test_cota = test_cota + sess_result[0]

                sess_result = sess.run([G_loss],feed_dict={x:current_batch_divide,y:current_batch_label_divide})
                print("Current Iter : ",iter,' current batch: ',batch_size_index ,' current divide index : ',divide_batch_index ,' Current cost: ', sess_result[0],end='\r')
                train_cota = train_cota + sess_result[0]

        # if it is print size print the cost and Sample Image
        if iter % print_size==0:
            print("\n--------------")   
            print('Current Iter: ',iter,' Accumulated Train cost : ', train_cota/(len(train_batch)/(batch_size)),end='\n')
            print('Current Iter: ',iter,' Accumulated Test cost : ', test_cota/(len(train_batch)/(batch_size)),end='\n')
            print("--------------")

        train_cot.append(train_cota/(len(train_batch)/(batch_size)))
        test_cot.append(test_cota/(len(test_batch)/(batch_size)))
        train_cota,train_acca = 0,0
        test_cota,test_acca = 0,0

    # Normalize the cost of the training
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))
    test_cot = (test_cot-min(test_cot) ) / (max(test_cot)-min(test_cot))

    # plot the training and testing graph
    plt.figure()
    plt.legend()
    plt.plot(range(len(train_cot)),train_cot,color='green',label='Train cost ovt')
    plt.plot(range(len(train_cot)),train_cot,color='red',label='Test cost ovt')
    plt.title("Average Accuracy / Cost Over Time")
    plt.savefig("viz/Case Train.png",bbox_inches='tight')
    plt.close('all')

    # generate final output value for test
    final_output = np.zeros_like(test_label)
    for batch_size_index in range(0,len(test_batch),batch_size):
        current_batch = test_batch[batch_size_index:batch_size_index+batch_size]
        current_batch_label = test_label[batch_size_index:batch_size_index+batch_size]
        for divide_batch_index in range(0,192,divide_size):
            current_batch_divide = current_batch[:,divide_batch_index:divide_batch_index+divide_size,:,:,:]
            sess_result = sess.run(layer4_g,feed_dict={x:current_batch_divide})
            final_output[:,divide_batch_index:divide_batch_index+divide_size,:,:,:] = sess_result

    np.save('test_data.npy',test_batch)
    np.save('test_label_gt.npy',test_label)
    np.save('test_label_pd.npy',final_output)





# -- end code --