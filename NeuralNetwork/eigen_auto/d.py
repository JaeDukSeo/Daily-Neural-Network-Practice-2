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
import matplotlib

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(6278)
tf.set_random_seed(6728)
ia.seed(6278)

# ===== Activation Function =====
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
# ===== Activation Function =====

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
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
train, test = tf.keras.datasets.mnist.load_data()
x_data, train_label, y_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
x_data = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
y_data = y_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
train_batch = np.zeros((55000,28,28,1))
test_batch = np.zeros((10000,28,28,1))
for x in range(len(x_data)):
    train_batch[x,:,:,:] = np.expand_dims(imresize(x_data[x,:,:,0],(28,28)),axis=3)
for x in range(len(y_data)):
    test_batch[x,:,:,:] = np.expand_dims(imresize(y_data[x,:,:,0],(28,28)),axis=3)

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

train_batch = train_batch/255.0
test_batch = test_batch/255.0
train_batch = train_batch[:10000]
train_label = train_label[:10000]

# hyper parameter 
num_epoch = 21
batch_size = 20
print_size = 2

learning_rate = 0.0003
learnind_rate_decay = 0.0
beta1,beta2,adam_e = 0.9,0.999,1e-8

# define class here
el1 = CNN(3,1,512)
el2 = CNN(3,512,512)
el3 = FNN(7*7*512,3,tf_tanh,d_tf_iden)

network_effect = tf.random_uniform([3,3],0,1)

dl1 = FNN(3,7*7*512,tf_tanh,d_tf_iden)
dl2 = CNN_Trans(3,512,512)
dl3 = CNN_Trans(3,256,512)
final_cnn = CNN(3,256,1,tf_sigmoid,d_tf_sigmoid)

# graph
x = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32,name="input")

# encoder
elayer1 = el1.feedforward(x,padding='SAME')
elayer2_input = tf.nn.avg_pool(elayer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
elayer2 = el2.feedforward(elayer2_input,padding='SAME')
elayer3_input = tf.nn.avg_pool(elayer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
elayer3_flatten = tf.reshape(elayer3_input,[batch_size,-1])
elayer3 = el3.feedforward(elayer3_flatten)

# co varience matrix and eign values
mean_data = tf.reduce_mean(elayer3,0)
center_matrix = elayer3 - mean_data
covarience_matrix = tf.matmul(tf.transpose(center_matrix),center_matrix) / batch_size   
covarience_matrix2 = tf.matmul(covarience_matrix,network_effect)
e_value,e_vector = tf.linalg.eigh(covarience_matrix2)
d_layer_input = tf.matmul(elayer3,e_vector)

# decoder
dlayer1 = dl1.feedforward(d_layer_input)
dlayer2_reshape = tf.reshape(dlayer1,[batch_size,7,7,512])
dlayer2 = dl2.feedforward(dlayer2_reshape,stride=2,padding='SAME')
dlayer3 = dl3.feedforward(dlayer2,stride=2,padding='SAME')
final_output = final_cnn.feedforward(dlayer3)

# calculate the loss
final_output_vec = tf.reshape(final_output,[batch_size,-1])
final_x = tf.reshape(x,[batch_size,-1])
cost1 = tf.reduce_mean(tf.square(final_output-x))
cost2 = -tf.reduce_sum(final_x * tf.log(1e-10 + final_output_vec)+ (1-final_x) * tf.log(1e-10 + 1 - final_output_vec))
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost1 + cost2)

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

        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            sess_result = sess.run([cost1,cost2,auto_train],feed_dict={x:current_batch})
            print("Current Iter : ",iter ," current batch: ",batch_size_index, ' Current cost: ', sess_result[0]+sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0] + sess_result[1]

        if iter % print_size==0:
            print("\n--------------")
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size)),end='\n')
            print("----------")

        if iter % 2 == 0:
            test_example = train_batch[:batch_size,:,:,:]
            sess_results = sess.run([final_output],feed_dict={x:test_example})

            random_choosen = np.random.randint(0,batch_size)
            sess_results = sess_results[0][random_choosen,:,:,:]
            test_example = test_example[random_choosen,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example),cmap='gray')
            plt.axis('off')
            plt.title('Original Image')
            plt.savefig('train_change/'+str(iter)+"a_Original_Image.png",bbox_inches='tight')

            plt.figure()
            plt.imshow(np.squeeze(sess_results).astype(np.float32),cmap='gray')
            plt.axis('off')
            plt.title('Generated Mask')
            plt.savefig('train_change/'+str(iter)+"b_Reconstructed_Image.png",bbox_inches='tight')
            plt.close('all')

        train_cot.append(train_cota/(len(train_batch)/(batch_size)))
        train_cota,train_acca = 0,0

    # Normalize the cost of the training
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))

    # plot the training and testing graph
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig("viz/Case Train.png")
    plt.close('all')

    # after all of the training generate the 3D plot figure
    test_batch = train_batch[:1000]
    test_label = train_label[:1000]

    test_latent = sess.run(elayer3,feed_dict={x:test_batch[:batch_size,:,:,:]})
    for iii in range(batch_size,len(test_batch),batch_size):
        temp = sess.run(elayer3,feed_dict={x:test_batch[iii:batch_size+iii,:,:,:]})
        test_latent = np.vstack((test_latent,temp))

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    color_dict = {
        0:'red',
        1:'blue',
        2:'green',
        3:'yellow',
        4:'purple',
        5:'grey',
        6:'black',
        7:'violet',
        8:'silver',
        9:'cyan',
    }

    color_mapping = [color_dict[x] for x in np.argmax(test_label,1) ]
    ax.scatter(test_latent[:,0], test_latent[:,1],test_latent[:,2],c=color_mapping,label=str(color_dict))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    ax.grid(True)
    plt.show()
    plt.close('all')
    
    # next show the transformed after co varience
    test_latent = sess.run(d_layer_input,feed_dict={x:test_batch[:batch_size,:,:,:]})
    for iii in range(batch_size,len(test_batch),batch_size):
        temp = sess.run(d_layer_input,feed_dict={x:test_batch[iii:batch_size+iii,:,:,:]})
        test_latent = np.vstack((test_latent,temp))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    color_dict = {
        0:'red',
        1:'blue',
        2:'green',
        3:'yellow',
        4:'purple',
        5:'grey',
        6:'black',
        7:'violet',
        8:'silver',
        9:'cyan',
    }

    color_mapping = [color_dict[x] for x in np.argmax(test_label,1) ]
    ax.scatter(test_latent[:,0], test_latent[:,1],test_latent[:,2],c=color_mapping,label=str(color_dict))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    ax.grid(True)
    plt.show()


# -- end code --