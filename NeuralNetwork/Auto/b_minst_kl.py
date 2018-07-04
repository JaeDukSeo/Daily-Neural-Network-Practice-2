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
from tensorflow.examples.tutorials.mnist import input_data

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(6278)
tf.set_random_seed(6728)
ia.seed(6278)

def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32)  + (tf_elu(tf.cast(tf.less_equal(x,0),tf.float32) * x) + 1.0)
def tf_softmax(x): return tf.nn.softmax(x)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

# ================= DATA AUGMENTATION =================
seq = iaa.Sequential([
    iaa.Sometimes(0.5,
        iaa.Fliplr(0.5), # horizontal flips
    ),
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Sometimes(0.5,
    iaa.Affine(
        rotate=(-180, 180),
    ))
], random_order=True) # apply augmenters in random order
# ================= DATA AUGMENTATION =================

# ================= LAYER CLASSES =================
class CNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.truncated_normal([k,k,inc,out],stddev=0.05))
        # self.w = tf.Variable(xavier_init_cnn(k,inc,out))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding) 
        self.layerA = tf_elu(self.layer)
        return self.layerA 

    def backprop(self,gradient,learning_rate_change,stride=1,padding='SAME'):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_elu(self.layer) 
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
        adam_middel = learning_rate_change/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat)  )))         

        return grad_pass,update_w 

class CNN_Trans():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.truncated_normal([k,k,inc,out],stddev=0.05))
        # self.w = tf.Variable(xavier_init_cnn(k,inc,out))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        output_shape2 = self.input.shape[2].value * stride
        self.layer  = tf.nn.conv2d_transpose(
            input,self.w,output_shape=[batch_size,output_shape2,output_shape2,self.w.shape[2].value],
            strides=[1,stride,stride,1],padding=padding) 
        self.layerA = tf_elu(self.layer)
        return self.layerA 

class FNN():
    def __init__(self,input_dim,hidden_dim):
        self.w = tf.Variable(tf.truncated_normal([input_dim,hidden_dim],stddev=0.05))
        # self.w = tf.Variable(xavier_init(input_dim,hidden_dim)) here

    def feedforward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = tf_elu(self.layer)
        return self.layerA

# ================= LAYER CLASSES =================

# data
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)

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

# hyper parameter
num_epoch = 101
batch_size = 100
print_size = 10

learning_rate = 0.0008
learnind_rate_decay = 0.0
beta1,beta2,adam_e = 0.9,0.999,1e-8

# define class here
l1 = CNN(3,1,16)
l2 = CNN(3,16,32)
l3_prep = FNN(7*7*32,100)

mean_vector,std_vector = FNN(100,3),FNN(100,3)

l4_prep = FNN(3,100)
l4 = FNN(100,7*7*32)
l5 = CNN_Trans(3,16,32)
l52 = CNN(3,16,16)
l6 = CNN_Trans(3,1,16)
l62 = CNN(3,1,1)

# graph
x = tf.placeholder(shape=[batch_size,28,28,1],dtype=tf.float32)

layer1 = l1.feedforward(x)
layer2_Input = tf.nn.avg_pool(layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2 = l2.feedforward(layer2_Input) 
layer3_Input = tf.nn.avg_pool(layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

layer3_prep_Input = tf.reshape(layer3_Input,[batch_size,-1])
layer3_prep = l3_prep.feedforward(layer3_prep_Input)

mean = mean_vector.feedforward(layer3_prep)
std  = std_vector.feedforward(layer3_prep)
samples = tf.random_normal([batch_size,3], 0, 1, dtype=tf.float32)
z = mean + tf.sqrt(tf.exp(std)) * samples

layer4_prep = l4_prep.feedforward(z)
layer4 = l4.feedforward(layer4_prep)

layer5_Input = tf.reshape(layer4,[batch_size,7,7,32])
layer5 = l5.feedforward(layer5_Input,stride=2)
layer52 = l52.feedforward(layer5)
layer6 = l6.feedforward(layer52,stride=2)
layer62 = l62.feedforward(layer6)

# x_vector = tf.reshape(x,[batch_size,-1])
# layer72_vector = tf.reshape(layer72,[batch_size,-1])
# reconstr_loss = -tf.reduce_sum( x_vector * tf.log(1e-10 +layer72_vector) + (1-x_vector) * tf.log(1e-10 + 1 - layer72_vector),axis=1 )
reconstr_loss2 = tf.reduce_mean(tf.square(layer62-x))
latent_loss = -0.5 * tf.reduce_sum(1 + std - tf.square(mean)  - tf.exp(std), 1)
cost = tf.reduce_mean(latent_loss+reconstr_loss2) 
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

        train_batch = shuffle(train_batch)

        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            sess_result = sess.run([cost,auto_train],feed_dict={x:current_batch})
            print("Current Iter : ",iter ," current batch: ",batch_size_index, ' Current cost: ', sess_result[0],end='\r')
            train_cota = train_cota + sess_result[0]

        if iter % print_size==0:
            print("\n--------------")
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size)),end='\n')
            print("----------")

        if iter % 2 == 0:
            test_example =    train_batch[:batch_size,:,:,:]
            test_example_gt = train_batch[:batch_size,:,:,:]
            sess_results = sess.run([layer62],feed_dict={x:test_example})

            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]
            test_example_gt = test_example_gt[0,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example),cmap='gray')
            plt.axis('off')
            plt.title('Original Image')
            plt.savefig('train_change/'+str(iter)+"a_Original_Image.png")

            sess_results[:,:,0] = (sess_results[:,:,0]-sess_results[:,:,0].min())/(sess_results[:,:,0].max()-sess_results[:,:,0].min())
            # sess_results[:,:,1] = (sess_results[:,:,1]-sess_results[:,:,1].min())/(sess_results[:,:,1].max()-sess_results[:,:,1].min())
            # sess_results[:,:,2] = (sess_results[:,:,2]-sess_results[:,:,2].min())/(sess_results[:,:,2].max()-sess_results[:,:,2].min())

            plt.figure()
            plt.imshow(np.squeeze(sess_results).astype(np.float32),cmap='gray')
            plt.axis('off')
            plt.title('Generated Mask')
            plt.savefig('train_change/'+str(iter)+"c_Generated_Mask.png")
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


# -- end code --