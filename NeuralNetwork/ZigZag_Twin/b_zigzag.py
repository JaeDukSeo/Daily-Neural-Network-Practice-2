import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from imgaug import augmenters as iaa
import imgaug as ia
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
plt.style.use('seaborn-dark')

np.random.seed(678) 
tf.set_random_seed(678)
ia.seed(678)

# activation
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32)  + ( tf_elu(tf.cast(tf.less_equal(x,0),tf.float32) * x) + 1.0)
def tf_soft(x): return tf.nn.softmax(x)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Different noises of different channels
def gaussian_noise_layer(input_layer,std=1.0):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + 0.1*noise

def possin_layer(layer):
    noise = tf.random_poisson(lam=1.0,shape=tf.shape(layer),dtype=tf.float32)
    return 0.1*noise + layer

def uniform_layer(input_layer):
    noise = tf.random_uniform(shape=tf.shape(input_layer),minval=0.5,dtype=tf.float32)
    return 0.6*noise + input_layer

# data aug
seq = iaa.Sequential([
    iaa.Fliplr(1.0), # horizontal flips
    iaa.Sometimes(0.1,
        iaa.GaussianBlur(sigma=(0,0.5))
    )
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

# class recurrent CNN and CNN
class Zig_Zag_RCNN():

    def __init__(self,timestamp,c_in,c_out,x_kernel,h_kernel,size):

        self.w_1 = tf.Variable(tf.random_normal([x_kernel,x_kernel,c_in,c_out]))
        self.h_1 = tf.Variable(tf.random_normal([h_kernel,h_kernel,c_out,c_out]))

        self.w_2 = tf.Variable(tf.random_normal([x_kernel,x_kernel,c_in,c_out]))
        self.h_2 = tf.Variable(tf.random_normal([h_kernel,h_kernel,c_out,c_out]))

        self.input_record_1   = tf.Variable(tf.zeros([timestamp,batch_size//2,size,size,c_in]))
        self.hidden_record_1  = tf.Variable(tf.zeros([timestamp+1,batch_size//2,size,size,c_out]))
        self.hiddenA_record_1 = tf.Variable(tf.zeros([timestamp+1,batch_size//2,size,size,c_out]))
        
        self.input_record_2   = tf.Variable(tf.zeros([timestamp,batch_size//2,size,size,c_in]))
        self.hidden_record_2  = tf.Variable(tf.zeros([timestamp+1,batch_size//2,size,size,c_out]))
        self.hiddenA_record_2 = tf.Variable(tf.zeros([timestamp+1,batch_size//2,size,size,c_out]))

    def feedforward_straight(self,input,timestamp):

        # assign the inputs 
        hidden_assign = []
        
        # perform feed forward on left
        layer_1 =  tf.nn.conv2d(input,self.w_1,strides=[1,1,1,1],padding='SAME')  + \
        tf.nn.conv2d(self.hiddenA_record_1[timestamp,:,:,:,:],self.h_1,strides=[1,1,1,1],padding='SAME') 
        layerA_1 = tf_elu(layer_1)

        # perform feed forward on right
        layer_2 =  tf.nn.conv2d(input,self.w_2,strides=[1,1,1,1],padding='SAME')  + \
        tf.nn.conv2d(self.hiddenA_record_2[timestamp,:,:,:,:],self.h_2,strides=[1,1,1,1],padding='SAME') 
        layerA_2 = tf_elu(layer_2)
    
        # assign for left
        hidden_assign.append(tf.assign(self.hidden_record_1[timestamp+1,:,:,:,:],layer_1))
        hidden_assign.append(tf.assign(self.hiddenA_record_1[timestamp+1,:,:,:,:],layerA_1))

        # assign for right
        hidden_assign.append(tf.assign(self.hidden_record_2[timestamp+1,:,:,:,:],layer_2))
        hidden_assign.append(tf.assign(self.hiddenA_record_2[timestamp+1,:,:,:,:],layerA_2))

        return layerA_1,layerA_2,hidden_assign

    def feedforward_zigzag(self,input,timestamp):
        
        # assign the inputs 
        hidden_assign = []
        
        # perform feed forward on left
        layer_1 =  tf.nn.conv2d(input,self.w_1,strides=[1,1,1,1],padding='SAME')  + \
        tf.nn.conv2d(self.hiddenA_record_2[timestamp,:,:,:,:],self.h_1,strides=[1,1,1,1],padding='SAME') 
        layerA_1 = tf_elu(layer_1)

        # perform feed forward on right
        layer_2 =  tf.nn.conv2d(input,self.w_2,strides=[1,1,1,1],padding='SAME')  + \
        tf.nn.conv2d(self.hiddenA_record_1[timestamp,:,:,:,:],self.h_2,strides=[1,1,1,1],padding='SAME') 
        layerA_2 = tf_elu(layer_2)
    
        # assign for left
        hidden_assign.append(tf.assign(self.hidden_record_1[timestamp+1,:,:,:,:],layer_1))
        hidden_assign.append(tf.assign(self.hiddenA_record_1[timestamp+1,:,:,:,:],layerA_1))

        # assign for right
        hidden_assign.append(tf.assign(self.hidden_record_2[timestamp+1,:,:,:,:],layer_2))
        hidden_assign.append(tf.assign(self.hiddenA_record_2[timestamp+1,:,:,:,:],layerA_2))

        return layerA_1,layerA_2,hidden_assign
        
    
class CNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerA = tf_elu(self.layer)
        return self.layerA 

    def backprop(self,gradient,learing_rate_dynamic):
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

        v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 

        def f1(): return v_t
        def f2(): return self.v_hat_prev

        v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
        adam_middel = learing_rate_dynamic/(tf.sqrt(v_max) + adam_e)
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

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

# hyper 
num_epoch = 21
batch_size = 50
print_size = 1
timestamp = 4

learning_rate = 0.00008
beta1,beta2,adam_e = 0.9,0.9,1e-8

# define class
ZigZag_RCNN = Zig_Zag_RCNN(timestamp=timestamp,c_in=3,c_out=96,x_kernel=3,h_kernel=1,size=32)

l1 = CNN(3,192,192)
l2 = CNN(3,192,192)
l3 = CNN(3,192,192)

l4 = CNN(3,192,192)
l5 = CNN(1,192,192)
l6 = CNN(1,192,10)

# graph 
x = tf.placeholder(shape=[batch_size,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[batch_size//2,10],dtype=tf.float32)

x_original_image = x[:batch_size//2,:,:,:]
x_augment_image  = x[batch_size//2:,:,:,:]

# at time stamp 0 perform 
layer1_L_ts_0,layer1_R_ts_0,layer1_ts_0_Up = ZigZag_RCNN.feedforward_straight(x_original_image,0)
layer1_L_ts_1,layer1_R_ts_1,layer1_ts_1_Up = ZigZag_RCNN.feedforward_zigzag(x_augment_image,1)
layer1_L_ts_2,layer1_R_ts_2,layer1_ts_2_Up = ZigZag_RCNN.feedforward_zigzag(x_original_image,2)
layer1_L_ts_3,layer1_R_ts_3,layer1_ts_3_Up = ZigZag_RCNN.feedforward_zigzag(x_augment_image,3)

# concat the final output of the two stream and give it to CNN
convolution_network_input = tf.concat([layer1_L_ts_3,layer1_R_ts_3],axis=3)

layer1_Input = tf.nn.avg_pool(convolution_network_input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer1 = l2.feedforward(layer1_Input)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)

layer4_Input = tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input)
layer5 = l5.feedforward(layer4)
layer6 = l6.feedforward(layer5)

final_global = tf.reduce_mean(layer6,[1,2])
final_soft = tf_soft(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y) )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate,beta2=0.9).minimize(cost)

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        # shuffle the training images for regul
        train_batch,train_label = shuffle(train_batch,train_label)

        for batch_size_index in range(0,len(train_batch),batch_size//2):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size//2]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size//2]

            # online data augmentation here and standard normalization
            images_aug = seq.augment_images(current_batch.astype(np.float32))
            current_batch = np.vstack((current_batch,images_aug)).astype(np.float32)
            # current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            # current_batch = np.insert(current_batch,range(1,len(current_batch)+1),images_aug,axis=0).astype(np.float32)
            # current_batch_label = np.insert(current_batch_label,range(1,len(current_batch_label)+1),current_batch_label,axis=0)
            current_batch[:,:,:,0]  = (current_batch[:,:,:,0] - current_batch[:,:,:,0].mean(axis=0)) / ( current_batch[:,:,:,0].std(axis=0)+1e-10)
            current_batch[:,:,:,1]  = (current_batch[:,:,:,1] - current_batch[:,:,:,1].mean(axis=0)) / ( current_batch[:,:,:,1].std(axis=0)+1e-10)
            current_batch[:,:,:,2]  = (current_batch[:,:,:,2] - current_batch[:,:,:,2].mean(axis=0)) / ( current_batch[:,:,:,2].std(axis=0)+1e-10)
            # current_batch,current_batch_label  = shuffle(current_batch,current_batch_label)
            # online data augmentation here and standard normalization
            
            sess_result = sess.run([cost,accuracy,auto_train,correct_prediction,final_soft],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]
            current_batch[:,:,:,0]  = (current_batch[:,:,:,0] - current_batch[:,:,:,0].mean(axis=0)) / ( current_batch[:,:,:,0].std(axis=0)+1e-10)
            current_batch[:,:,:,1]  = (current_batch[:,:,:,1] - current_batch[:,:,:,1].mean(axis=0)) / ( current_batch[:,:,:,1].std(axis=0)+1e-10)
            current_batch[:,:,:,2]  = (current_batch[:,:,:,2] - current_batch[:,:,:,2].mean(axis=0)) / ( current_batch[:,:,:,2].std(axis=0)+1e-10)
            sess_result = sess.run([cost,accuracy,final_soft],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n---------- " )
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size//2)),' Current Acc: ', train_acca/(len(train_batch)/(batch_size//2) ),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/(batch_size//2)))
        train_cot.append(train_cota/(len(train_batch)/(batch_size//2)))
        test_acc.append(test_acca/(len(test_batch)/batch_size))
        test_cot.append(test_cota/(len(test_batch)/batch_size))
        test_cota,test_acca = 0,0
        train_cota,train_acca = 0,0

    # normalize the cost
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))
    test_cot = (test_cot-min(test_cot) ) / (max(test_cot)-min(test_cot))

    # training done
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig('case b train.png')

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig('case b test.png')

# -- end code --