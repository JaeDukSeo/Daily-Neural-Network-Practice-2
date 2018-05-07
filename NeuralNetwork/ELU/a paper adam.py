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
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32)  + ( tf_elu(tf.cast(tf.less_equal(x,0),tf.float32) * x) + 1.0)
def tf_softmax(x): return tf.nn.softmax(x)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# class
class CNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05))
        self.B = tf.Variable(tf.random_uniform([k,k,inc,out],minval=-0.5,maxval=0.5))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerA = tf_elu(self.layer)
        return self.layerA

    def backprop(self,gradient,feedback=False):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_elu(self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        if feedback:
            grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
            filter= self.B,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
            )       
        else:
            grad_pass = tf.nn.conv2d_backprop_input(
                input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
                filter= self.w,out_backprop = grad_middle,
                strides=[1,1,1,1],padding='SAME'
            )

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))
        
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)

        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,grad_update  

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
num_epoch = 1001
batch_size = 100
print_size = 5
learning_rate = 0.00001
beta1,beta2,adam_e = 0.9,0.999,1e-8

proportion_rate = 1
decay_rate = 0.05

# define class
l1 = CNN(5,3,192)

l2 = CNN(1,192,192)
l3 = CNN(3,192,240)

l4 = CNN(1,240,240)
l5 = CNN(2,240,260)

l6 = CNN(1,260,260)
l7 = CNN(2,260,280)

l8 = CNN(1,280,280)
l9 = CNN(2,280,300)

l10 = CNN(1,300,300)
l11 = CNN(1,300,10)

# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
decay_dilated_rate = proportion_rate / (1 + decay_rate * iter_variable)

layer1 = l1.feedforward(x)

layer2_Input = tf.nn.max_pool(layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2 = l2.feedforward(layer2_Input)
layer3 = l3.feedforward(layer2)
layer3 = tf.nn.dropout(layer3,0.9)

layer4_Input = tf.nn.max_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input)
layer5 = l5.feedforward(layer4)
layer5 = tf.nn.dropout(layer5,0.8)

layer6_Input = tf.nn.max_pool(layer5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer6 = l6.feedforward(layer6_Input)
layer7 = l7.feedforward(layer6)
layer7 = tf.nn.dropout(layer7,0.7)

layer8_Input = tf.nn.max_pool(layer7,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer8 = l8.feedforward(layer8_Input)
layer9 = l9.feedforward(layer8)
layer9 = tf.nn.dropout(layer9,0.6)

layer10_Input = tf.nn.max_pool(layer9,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer10 = l10.feedforward(layer10_Input)
layer10 = tf.nn.dropout(layer10,0.5)

layer11 = l11.feedforward(layer10)

final_reshape = tf.reshape(layer11,[batch_size,-1])
final_soft = tf_softmax(final_reshape)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_reshape,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(cost)

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

            sess_result = sess.run([cost,accuracy,auto_train,correct_prediction,final_soft,final_reshape],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter})
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

    # training done
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.show()

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.show()





# -- end code --