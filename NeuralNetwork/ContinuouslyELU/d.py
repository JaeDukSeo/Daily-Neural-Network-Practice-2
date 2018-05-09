import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize

def tf_celu(x,alpha=2.0 ):
    mask_greater = tf.cast(tf.greater_equal(x,0),tf.float32) * x
    mask_smaller = tf.cast(tf.less(x,0),tf.float32) * x
    middle = alpha * ( tf.exp(tf.divide(mask_smaller,alpha)) - 1)
    return middle + mask_greater
def d_tf_celu(x,alpha=2.0):
    mask_greater = tf.cast(tf.greater_equal(x,0),tf.float32) 
    mask_smaller = tf.cast(tf.less(x,0),tf.float32) * x
    middle  =  tf.exp(tf.divide(mask_smaller,alpha))
    return middle + mask_greater

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
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerA = tf_celu(self.layer)
        return self.layerA 


    def backprop(self,gradient):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_celu(self.layer) 
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

        v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 

        def f1(): return v_t
        def f2(): return self.v_hat_prev

        v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
        adam_middel = learning_rate/(tf.sqrt(v_max) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,self.m))))
        update_w.append(
            tf.assign( self.v_prev,v_t )
        )
        update_w.append(
            tf.assign( self.v_hat_prev,v_max )
        )        
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

# hyper
num_epoch = 51
batch_size = 100
print_size = 2
learning_rate = 0.0008
beta1,beta2,adam_e = 0.9,0.9,1e-8

# proportion_rate = 0.0001
# decay_rate = 10
proportion_rate = 1
decay_rate = 0.1

# define class
l1 = CNN(5,3,200)
l2 = CNN(3,200,200)
l3 = CNN(1,200,200)
l4 = CNN(3,200,200)
l5 = CNN(1,200,200)
l6 = CNN(2,200,200)
l7 = CNN(1,200,10)

# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
decay_dilated_rate = proportion_rate / (1 + decay_rate * iter_variable)

layer1 = l1.feedforward(x)

layer2 = l2.feedforward(layer1)

layer3_Input = tf.nn.avg_pool(layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3 = l3.feedforward(layer3_Input)

layer4_Input = tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input)

layer5_Input = tf.nn.avg_pool(layer4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer5 = l5.feedforward(layer5_Input)

layer6_Input = tf.nn.avg_pool(layer5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer6 = l6.feedforward(layer6_Input)

layer7_Input = tf.nn.avg_pool(layer6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer7 = l7.feedforward(layer7_Input)

final_reshape = tf.reshape(layer7,[batch_size,-1])
final_soft = tf_softmax(final_reshape)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_reshape,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

grad7,grad7_up = l7.backprop(tf.reshape(final_soft-y,[batch_size,1,1,10] ))

grad6_Input = tf_repeat(grad7,[1,2,2,1])
grad6,grad6_up = l6.backprop(grad6_Input  )

grad7_dialted = tf_repeat(grad7,[1,4,4,1])
grad5_Input = tf_repeat(grad6,[1,2,2,1])
grad5,grad5_up = l5.backprop(grad5_Input+ decay_dilated_rate * (grad7_dialted) )

grad6_dialted = tf_repeat(grad6,[1,4,4,1])
grad4_Input = tf_repeat(grad5,[1,2,2,1])
grad4,grad4_up = l4.backprop(grad4_Input+ decay_dilated_rate * (grad6_dialted) )

grad5_dialted = tf_repeat(grad5,[1,4,4,1])
grad3_Input = tf_repeat(grad4,[1,2,2,1])
grad3,grad3_up = l3.backprop(grad3_Input+ decay_dilated_rate * (grad5_dialted) )

grad4_dialted = tf_repeat(grad4,[1,4,4,1])
grad2_Input = tf_repeat(grad3,[1,2,2,1])
grad2,grad2_up = l2.backprop(grad2_Input+ decay_dilated_rate * (grad4_dialted) )

grad1,grad1_up = l1.backprop(grad2 )

grad_update = grad7_up + grad6_up + grad5_up + grad4_up + \
                grad3_up + grad2_up + grad1_up


# # sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)

        for batch_size_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size]
            sess_result = sess.run([cost,accuracy,grad_update,final_soft,final_reshape],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter})
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
            print("\n----------")
            print('Train Current cost: ', train_cota/(len(train_batch)/batch_size),' Current Acc: ', train_acca/(len(train_batch)/batch_size),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/batch_size))
        train_cot.append(train_cota/(len(train_batch)/batch_size))
        test_acc.append(test_acca/(len(test_batch)/batch_size))
        test_cot.append(test_cota/(len(test_batch)/batch_size))
        test_cota,test_acca = 0,0
        train_cota,train_acca = 0,0
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))
    test_cot = (test_cot-min(test_cot) ) / (max(test_cot)-min(test_cot))
    # training done
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig("Case d Train.png")
    

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("Case d Test.png")





# -- end code --