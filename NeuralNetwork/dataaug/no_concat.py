import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from tensorflow.examples.tutorials.mnist import input_data
from imgaug import augmenters as iaa
import imgaug as ia

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(678) 
tf.set_random_seed(678)
ia.seed(678)

def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32)  + ( tf_elu(tf.cast(tf.less_equal(x,0),tf.float32) * x) + 1.0)

def tf_tanh(x): return tf.nn.tanh(x)
def tf_relu(x): return tf.nn.relu(x)


def tf_softmax(x): return tf.nn.softmax(x)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# data aug
seq = iaa.Sequential([
    iaa.Fliplr(1.0), # horizontal flips
    iaa.Affine(
        scale={"x": (1.0, 1.2), "y": (1.0, 1.2)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-15, 15),
        shear=(-4, 4)
    ),
], random_order=True) # apply augmenters in random order

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

# class
class CNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.truncated_normal([k,k,inc,out],stddev=0.05))
        self.b = tf.Variable(tf.truncated_normal([out],stddev=0.05))

        # self.w2 = tf.Variable(tf.truncated_normal([k,k,out,out],stddev=0.05))
        # self.b2 = tf.Variable(tf.truncated_normal([out],stddev=0.05))
        
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding)+self.b
        mean,var = tf.nn.moments(self.layer,axis=0)
        self.layer = tf.nn.batch_normalization(self.layer,scale=True,offset=True,mean=mean,variance=var,variance_epsilon=1e-8)
        self.layerA = tf_elu(self.layer) 
        return  self.layerA 

        # self.layerA = tf.concat([tf_elu(self.layer[:,:,:,:39]),
        #                         tf_elu(-1.0*self.layer[:,:,:,39:78]) ,
        #                         -1.0*tf_elu(self.layer[:,:,:,78:117]) ,
        #                         -1.0*tf_elu(-1.0*self.layer[:,:,:,117:]) 
        #                         ],axis=3)

        # Bias after Activation || Bias before Activation || No Bias

        # return  self.layerA +  0.9 * tf.nn.sigmoid(self.layerA) # 36/50 ||
        # return  self.layerA +  0.9 * tf.nn.tanh(self.layerA) # 45/61 ||
        # return  self.layerA +  0.9 * tf.atan(self.layerA) # 45/61 ||
        # return  self.layerA +  0.9 * tf.erf(self.layerA) # 45/60 ||
        
        # return  self.layerA +  0.9 * tf.nn.sigmoid(self.layerA) *self.layerA # 39/58 ||
        # return  self.layerA +  0.9 * tf.nn.tanh(self.layerA) *self.layerA # 41/56 ||
        # return  self.layerA +  0.9 * tf.atan(self.layerA) *self.layerA #
        # return  self.layerA +  0.9 * tf.erf(self.layerA) *self.layerA # 39/54 ||

        # return  self.layerA +  0.9 * tf.nn.sigmoid(self.layerA) *self.layer # 
        # return  self.layerA +  0.9 * tf.nn.tanh(self.layerA) * self.layer # 46/63 || || 47/67 with 50 
        # return  self.layerA +  0.9 * tf.atan(self.layerA) * self.layer # 45/63 || 44/63 || 45/64 , 48/66 with 50
        # return  self.layerA +  0.9 * tf.erf(self.layerA) *self.layer # 

        # return  self.layerA +  0.9 * tf.nn.sigmoid(self.layer) #
        # return  self.layerA +  0.9 * tf.nn.tanh(self.layer) # 45/60
        # return  self.layerA +  0.9 * tf.atan(self.layer) #
        # return  self.layerA +  0.9 * tf.erf(self.layer)  #

        # return  self.layerA +  0.9 * tf.nn.sigmoid(self.layerA) *self.layer # 
        # return  self.layerA +  0.9 * tf.nn.tanh(self.layer) * self.layerA # 39/56
        # return  self.layerA +  0.9 * tf.atan(self.layerA) *self.layer #
        # return  self.layerA +  0.9 * tf.erf(self.layerA) *self.layer #         
        
    def backprop(self,gradient,learning_rate_change,stride=1,padding='SAME'):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_elu(self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,stride,stride,1],padding=padding
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
            filter= self.w,out_backprop = grad_middle,
            strides=[1,stride,stride,1],padding=padding
        )

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   ))
        v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 
        def f1(): return v_t
        def f2(): return self.v_hat_prev
        v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
        adam_middel = learning_rate_change/(tf.sqrt(v_max) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,self.m))))
        update_w.append(tf.assign( self.v_prev,v_t ))
        update_w.append(tf.assign( self.v_hat_prev,v_max ))        

        return grad_pass,update_w   

class batch_norm():
    
    def __init__(self):
        self.gamma = tf.Variable(tf.random_normal([],stddev=0.05))
        self.beta = tf.Variable(tf.random_normal([],stddev=0.05))

        self.gamma = None
        self.xmu = None
        self.ivar = None
        self.sqrtvar = None
        self.var = None

    def feedforward(self,input):
        N,D = input.shape[0],input.shape[3]
        x = input 

        mu = 1.0/N * tf.reduce_sum(x,axis=0)
        self.xmu = x - mu
        sq  = self.xmu ** 2

        self.var = 1.0/N * tf.reduce_sum(sq,axis=0)
        self.sqrtvar = tf.sqrt(self.var + 1e-8)
        self.ivar = 1.0/self.sqrtvar

        self.xhat = self.xmu * self.ivar

        gammax = self.gamma * self.xhat + self.beta
        return gammax
        
# # data
PathDicom = "../../Dataset/cifar-10-batches-py/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if not ".html" in filename.lower() and not  ".meta" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# Read the data traind and Test
batch0 = unpickle(lstFilesDCM[0])
batch1 = unpickle(lstFilesDCM[1])
batch2 = unpickle(lstFilesDCM[2])
batch3 = unpickle(lstFilesDCM[3])
batch4 = unpickle(lstFilesDCM[4])

onehot_encoder = OneHotEncoder(sparse=True)
train_batch = np.vstack((batch0[b'data'],batch1[b'data'],batch2[b'data'],batch3[b'data'],batch4[b'data']))
train_label = np.expand_dims(np.hstack((batch0[b'labels'],batch1[b'labels'],batch2[b'labels'],batch3[b'labels'],batch4[b'labels'])).T,axis=1).astype(np.float32)
train_label = onehot_encoder.fit_transform(train_label).toarray().astype(np.float32)

test_batch = unpickle(lstFilesDCM[5])[b'data']
test_label = np.expand_dims(np.array(unpickle(lstFilesDCM[5])[b'labels']),axis=0).T.astype(np.float32)
test_label = onehot_encoder.fit_transform(test_label).toarray().astype(np.float32)

# reshape data
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))

# rotate data
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

# Normalize the image range from 0 to 1
# train_batch = train_batch / 255.0
test_batch = test_batch / 255.0

# hyper
num_epoch = 21
batch_size = 50
print_size = 1
beta1,beta2,adam_e = 0.9,0.9,1e-8

learning_rate = 0.0003
learning_rate_decay = 0.0001

proportion_rate = 1
decay_rate = 0.05

# define class
b1 = batch_norm()
b2 = batch_norm()
b3 = batch_norm()

channel_size = 196
l0 = CNN(3,3,channel_size//2)

l1 = CNN(3,channel_size//2,channel_size//2)
l2 = CNN(3,channel_size//2,channel_size//2)
l3 = CNN(3,channel_size//2,channel_size)

l4 = CNN(3,channel_size,channel_size)
l5 = CNN(3,channel_size,channel_size)
l6 = CNN(3,channel_size,channel_size)

l7 = CNN(3,channel_size,channel_size)
l8 = CNN(1,channel_size,channel_size)
l9 = CNN(1,channel_size,10)

l0w = l0.getw()
l1w,l2w,l3w = l1.getw(),l2.getw(),l3.getw()
l4w,l5w,l6w = l4.getw(),l5.getw(),l6.getw()
l7w,l8w,l9w = l7.getw(),l8.getw(),l9.getw()

# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate * (1.0/(1.0+learning_rate_decay*iter_variable))
decay_dilated_rate = proportion_rate * (1.0/(1.0+decay_rate*iter_variable))

layer0 = l0.feedforward(x)
layer0 = tf.nn.dropout(layer0,0.5)

layer1 = l1.feedforward(layer0)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)
layer3 = tf.nn.dropout(layer3,0.5)

layer4_Input = 0.5*tf.nn.max_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID') + \
0.5*tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input)
layer5 = l5.feedforward(layer4)
layer6 = l6.feedforward(layer5)
layer6 = tf.nn.dropout(layer6,0.5)

layer7_Input = 0.5*tf.nn.max_pool(layer6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID') + \
0.5*tf.nn.avg_pool(layer6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer7 = l7.feedforward(layer7_Input)
layer8 = l8.feedforward(layer7,padding='VALID')
layer9 = l9.feedforward(layer8,padding='VALID')

final_global = tf.reduce_mean(layer9,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate_change,beta2=0.9).minimize(cost)

# grad_prepare = tf.reshape(final_soft-y,[batch_size,1,1,10])
# grad9,grad9_up = l9.backprop(grad_prepare,learning_rate_change=learning_rate_change,padding='VALID')
# grad8,grad8_up = l8.backprop(grad9,learning_rate_change=learning_rate_change,padding='VALID')
# grad7,grad7_up = l7.backprop(grad8,learning_rate_change=learning_rate_change)

# grad6_Input = tf_repeat(grad7,[1,2,2,1])
# grad6,grad6_up = l6.backprop(grad6_Input,learning_rate_change=learning_rate_change)
# grad5,grad5_up = l5.backprop(grad6,learning_rate_change=learning_rate_change)
# grad4,grad4_up = l4.backprop(grad5,learning_rate_change=learning_rate_change)

# grad3_Input = tf_repeat(grad4,[1,2,2,1])
# grad3,grad3_up = l3.backprop(grad3_Input,learning_rate_change=learning_rate_change)
# grad2,grad2_up = l2.backprop(grad3,learning_rate_change=learning_rate_change)
# grad1,grad1_up = l1.backprop(grad2,learning_rate_change=learning_rate_change)

# grad_update = grad9_up + grad8_up+ grad7_up + grad6_up + grad5_up + grad4_up + grad3_up + grad2_up + grad1_up

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)
        test_batch,test_label   = shuffle(test_batch,test_label)
        for batch_size_index in range(0,len(train_batch),batch_size):
            
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size]

            # online data augmentation here and standard normalization
            images_aug = seq.augment_images(current_batch.astype(np.float32))
            current_batch = np.vstack((current_batch,images_aug)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch,current_batch_label = shuffle(current_batch,current_batch_label)
            current_batch = current_batch/255.0
            # online data augmentation here and standard normalization

            sess_result = sess.run([cost,accuracy,auto_train],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]
            sess_result = sess.run([cost,accuracy,correct_prediction],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n---------- LR : ", learning_rate * (1.0/(1.0+learning_rate_decay*iter)) )
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size)),' Current Acc: ', train_acca/(len(train_batch)/(batch_size)),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/(batch_size)))
        train_cot.append(train_cota/(len(train_batch)/(batch_size)))
        test_acc.append(test_acca/(len(test_batch)/batch_size))
        test_cot.append(test_cota/(len(test_batch)/batch_size))
        test_cota,test_acca = 0,0
        train_cota,train_acca = 0,0

    # Normalize the cost of the training
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))
    test_cot = (test_cot-min(test_cot) ) / (max(test_cot)-min(test_cot))

    # training done now plot
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig("Case Train.png")

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("Case Test.png")





# -- end code --