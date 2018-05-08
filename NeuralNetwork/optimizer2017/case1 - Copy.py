import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize

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

# class
class CNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.truncated_normal([k,k,inc,out],stddev=0.05))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
    def getw(self): return self.w
    def feedforward(self,input):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerA = tf_elu(self.layer)
        return self.layerA

    def backprop(self,gradient,weight_decay_tf,feedback=True):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_elu(self.layer) 
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

        update_w = []
        update_w.append(
            tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   )
        )
        update_w.append(
            tf.assign( self.v,self.v*beta2 + (1-beta2) * grad ** 2   )
        )
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel  = learning_rate/(tf.sqrt(v_hat) + adam_e)
        adam_middel2 = tf.subtract(tf.multiply(adam_middel,m_hat),learning_rate * tf.multiply(weight_decay_tf,self.w))
        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel2)))
        
        return grad_pass,update_w  

# data
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

# Normalize data from 0 to 1 per each channel
train_batch[:,:,:,0]  = (train_batch[:,:,:,0] - train_batch[:,:,:,0].min(axis=0)) / (train_batch[:,:,:,0].max(axis=0) - train_batch[:,:,:,0].min(axis=0))
train_batch[:,:,:,1]  = (train_batch[:,:,:,1] - train_batch[:,:,:,1].min(axis=0)) / (train_batch[:,:,:,1].max(axis=0) - train_batch[:,:,:,1].min(axis=0))
train_batch[:,:,:,2]  = (train_batch[:,:,:,2] - train_batch[:,:,:,2].min(axis=0)) / (train_batch[:,:,:,2].max(axis=0) - train_batch[:,:,:,2].min(axis=0))

test_batch[:,:,:,0]  = (test_batch[:,:,:,0] - test_batch[:,:,:,0].min(axis=0)) / (test_batch[:,:,:,0].max(axis=0) - test_batch[:,:,:,0].min(axis=0))
test_batch[:,:,:,1]  = (test_batch[:,:,:,1] - test_batch[:,:,:,1].min(axis=0)) / (test_batch[:,:,:,1].max(axis=0) - test_batch[:,:,:,1].min(axis=0))
test_batch[:,:,:,2]  = (test_batch[:,:,:,2] - test_batch[:,:,:,2].min(axis=0)) / (test_batch[:,:,:,2].max(axis=0) - test_batch[:,:,:,2].min(axis=0))

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)
print('------------------')

# hyper
num_epoch = 201
batch_size = 100
print_size = 5
learning_rate = 0.0001
beta1,beta2,adam_e = 0.9,0.999,1e-8
weight_decay = 0.512

proportion_rate = 1
decay_rate = 0.05

# define class
l1 = CNN(5,3,192)

l2 = CNN(1,192,192)
l3 = CNN(3,192,192)

l4 = CNN(1,192,240)
l5 = CNN(2,240,240)

l6 = CNN(1,240,260)
l7 = CNN(2,260,260)

l8 = CNN(1,260,280)
l9 = CNN(2,280,280)

l10 = CNN(1,280,300)
l11 = CNN(1,300,10)

# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

weight_decay_tf = tf.placeholder(tf.float32, shape=())
iter_variable = tf.placeholder(tf.float32, shape=())
decay_dilated_rate = proportion_rate / (1 + decay_rate * iter_variable)

layer1 = l1.feedforward(x)

layer2_Input = tf.nn.avg_pool(layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2 = l2.feedforward(layer2_Input)
layer3 = l3.feedforward(layer2)

layer4_Input = tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input)
layer5 = l5.feedforward(layer4)

layer6_Input = tf.nn.avg_pool(layer5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer6 = l6.feedforward(layer6_Input)
layer7 = l7.feedforward(layer6)

layer8_Input = tf.nn.avg_pool(layer7,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer8 = l8.feedforward(layer8_Input)
layer9 = l9.feedforward(layer8)

layer10_Input = tf.nn.avg_pool(layer9,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer10 = l10.feedforward(layer10_Input)

layer11 = l11.feedforward(layer10)

final_reshape = tf.reshape(layer11,[batch_size,-1])
final_soft = tf_softmax(final_reshape)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_reshape,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

grad11,grad11_up = l11.backprop(tf.reshape(final_soft-y,[batch_size,1,1,10] ),weight_decay_tf=weight_decay_tf)
grad10,grad10_up = l10.backprop(grad11,weight_decay_tf=weight_decay_tf)

grad9_Input = tf_repeat(grad10,[1,2,2,1])
grad9,grad9_up = l9.backprop(grad9_Input,weight_decay_tf=weight_decay_tf)
grad8,grad8_up = l8.backprop(grad9,weight_decay_tf=weight_decay_tf)

grad7_Input = tf_repeat(grad8,[1,2,2,1])
grad7,grad7_up = l7.backprop(grad7_Input,weight_decay_tf=weight_decay_tf)
grad6,grad6_up = l6.backprop(grad7,weight_decay_tf=weight_decay_tf)

grad5_Input = tf_repeat(grad6,[1,2,2,1])
grad5,grad5_up = l5.backprop(grad5_Input,weight_decay_tf=weight_decay_tf)
grad4,grad4_up = l4.backprop(grad5,weight_decay_tf=weight_decay_tf)

grad3_Input = tf_repeat(grad4,[1,2,2,1])
grad3,grad3_up = l3.backprop(grad3_Input,weight_decay_tf=weight_decay_tf)
grad2,grad2_up = l2.backprop(grad3,weight_decay_tf=weight_decay_tf)

grad1_Input = tf_repeat(grad2,[1,2,2,1])
grad1,grad1_up = l1.backprop(grad1_Input,weight_decay_tf=weight_decay_tf)
grad_update = grad11_up + grad10_up + grad9_up + grad8_up + \
                grad7_up + grad6_up + grad5_up + grad4_up + \
                grad3_up + grad2_up + grad1_up

# # sess
with tf.Session( ) as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        if iter == 10 : weight_decay = weight_decay * 0.5
        if iter == 50 : weight_decay = weight_decay * 0.5
        if iter == 100 : weight_decay = weight_decay * 0.5
        if iter == 150 : weight_decay = weight_decay * 0.5

        train_batch,train_label = shuffle(train_batch,train_label)

        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size]

            sess_result = sess.run([cost,accuracy,grad_update,correct_prediction,final_soft,final_reshape],feed_dict={x:current_batch,
            y:current_batch_label,iter_variable:iter,weight_decay_tf:weight_decay})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]

            sess_result = sess.run([cost,accuracy,final_soft,final_reshape],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n---------- Weight Decay: ",weight_decay)
            print('Train Current cost: ', train_cota/(len(train_batch)/batch_size),' Current Acc: ', train_acca/(len(train_batch)/batch_size),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/batch_size))
        train_cot.append(train_cota/(len(train_batch)/batch_size))
        test_acc.append(test_acca/(len(test_batch)/batch_size))
        test_cot.append(test_cota/(len(test_batch)/batch_size))
        test_cota,test_acca = 0,0
        train_cota,train_acca = 0,0

    # training done
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig('case 1 train.png')

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig('case 1 test.png')





# -- end code --