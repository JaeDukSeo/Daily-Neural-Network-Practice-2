import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from imgaug import augmenters as iaa
import imgaug as ia

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(678)
tf.set_random_seed(678)
ia.seed(678)

def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater_equal(x,0),tf.float32)  + (tf_elu(tf.cast(tf.less(x,0),tf.float32) * x) + 1.0)
def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(x): return tf.cast(tf.greater_equal(x,0),tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# ===== Rules that I have learned =====
# 1. Data augmentation horizontal flip
# 2. kernel size is 3 
# 3. LR decay is also good 
# 4. 3 block is good
# ===== Rules that I have learned =====

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
    
    def __init__(self,k,inc,out,act,d_act):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act
    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding)
        self.layerA = self.act(self.layer)
        return self.layerA 

    def backprop(self,gradient,learning_rate_change,batch_size_dynamic,stride=1,padding='SAME',adam=False,awsgrad=False,reg=False):
        grad_part_1 = gradient 
        grad_part_2 = self.d_act(self.layer) 
        grad_part_3 = self.input
        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,stride,stride,1],padding=padding
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size_dynamic] + list(grad_part_3.shape[1:]),
            filter= self.w,out_backprop = grad_middle,
            strides=[1,stride,stride,1],padding=padding
        )

        update_w = []
        if  adam:
            update_w.append(
                tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   )
            )
            update_w.append(
                tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   )
            )
            m_hat = self.m / (1-beta1)
            v_hat = self.v_prev / (1-beta2)
            adam_middel = learning_rate_change/(tf.sqrt(v_hat) + adam_e)
            adam_middel = tf.multiply(adam_middel,m_hat)
            if reg: adam_middel = adam_middel - learning_rate_change * decouple_reg * self.w
            update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel  )) )    

        if awsgrad:
            update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   ))
            v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 
            def f1(): return v_t
            def f2(): return self.v_hat_prev
            v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
            adam_middel = tf.multiply(learning_rate_change/(tf.sqrt(v_max) + adam_e),self.m)
            if reg: adam_middel = adam_middel - learning_rate_change * decouple_reg * self.w
            update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel  )  ))
            update_w.append(tf.assign( self.v_prev,v_t ))
            update_w.append(tf.assign( self.v_hat_prev,v_max ))        

        return grad_pass,update_w  

# data
PathDicom = "../../../Dataset/cifar-10-batches-py/"
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
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))

# reshape data rotate data
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2))
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)

# standardize Normalize data per channel
test_batch[:,:,:,0]  = (test_batch[:,:,:,0] - test_batch[:,:,:,0].mean(axis=0)) / ( test_batch[:,:,:,0].std(axis=0))
test_batch[:,:,:,1]  = (test_batch[:,:,:,1] - test_batch[:,:,:,1].mean(axis=0)) / ( test_batch[:,:,:,1].std(axis=0))
test_batch[:,:,:,2]  = (test_batch[:,:,:,2] - test_batch[:,:,:,2].mean(axis=0)) / ( test_batch[:,:,:,2].std(axis=0))

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

# hyper parameter
num_epoch = 21  
batch_size = 50
print_size = 1
beta1,beta2,adam_e = 0.9,0.9,1e-9

learning_rate = 0.0002
learnind_rate_decay = 0
decouple_reg  = 0.00001

proportion_rate = 0.00001
decay_rate = 0

channel_size = 156
l0 = CNN(3,3,channel_size,tf_elu,d_tf_elu)

l1_a = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l1 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l2 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l3 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l4 = CNN(1,channel_size,channel_size,tf_elu,d_tf_elu)

l5_a = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l5 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l6 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l7 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l8 = CNN(1,channel_size,channel_size,tf_elu,d_tf_elu)

l9_a = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l9 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l10 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l11 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l12 = CNN(1,channel_size,channel_size,tf_elu,d_tf_elu)

l13 = CNN(1,channel_size,10,tf_elu,d_tf_elu)

# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

batch_size_dynamic= tf.placeholder(tf.int32, shape=())

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_dynamic  = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate_dynamic * (1.0/(1.0+learnind_rate_decay*iter_variable))
decay_dilated_rate   = proportion_rate       * (1.0/(1.0+decay_rate*iter_variable))

droprate1 = tf.placeholder(tf.float32, shape=())
droprate2 = tf.placeholder(tf.float32, shape=())
droprate3 = tf.placeholder(tf.float32, shape=())

layer0 = l0.feedforward(x)

layer1a = l1_a.feedforward(layer0)
layer1 = l1.feedforward(layer1a)
layer2 = l2.feedforward(layer1 + decay_dilated_rate * layer1a,stride=2)
layer3 = l3.feedforward(layer2)
layer4 = l4.feedforward(layer3+ decay_dilated_rate * layer2)

layer5_Input = tf.nn.avg_pool(layer4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
layer5a = l5_a.feedforward(layer5_Input)
layer5 = l5.feedforward(layer5a) # here
layer6 = l6.feedforward(layer5+ decay_dilated_rate * (layer5a))
layer7 = l7.feedforward(layer6+ decay_dilated_rate * (layer5a+layer5))
layer8 = l8.feedforward(layer7+ decay_dilated_rate * (layer5a+layer5+layer6))

layer9_Input = tf.nn.avg_pool(layer8,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
layer9a = l9_a.feedforward(layer9_Input)
layer9  = l9.feedforward(layer9a) # here
layer10 = l10.feedforward(layer9+ decay_dilated_rate * (layer9a),stride=2)
layer11 = l11.feedforward(layer10)
layer12 = l12.feedforward(layer11+ decay_dilated_rate * (layer10))

layer13 = l13.feedforward(layer12)

final_global = tf.reduce_mean(layer13,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y) )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ==== manual back prop ====
grad_prepare = tf.reshape(final_soft-y,[batch_size_dynamic,1,1,10])
grad13,grad13_up = l13.backprop(grad_prepare,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True,reg=True)

grad12,grad12_up = l12.backprop(grad13,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True)
grad11,grad11_up = l11.backprop(grad12+decay_dilated_rate*grad13,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,adam=True,reg=True)
grad10,grad10_up = l10.backprop(grad11+decay_dilated_rate*(grad13 + grad12),learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True,reg=True,stride=2)
grad9,grad9_up = l9.backprop(grad10,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,adam=True,reg=True)
grad9_a,grad9_a_up = l9_a.backprop(grad9 + decay_dilated_rate * grad10,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True)

grad8_Input = tf_repeat(grad9_a,[1,2,2,1])
grad8,grad8_up = l8.backprop(grad8_Input,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,adam=True)
grad7,grad7_up = l7.backprop(grad8+decay_dilated_rate*grad8_Input,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True,reg=True)
grad6,grad6_up = l6.backprop(grad7+decay_dilated_rate*(grad8_Input + grad8),learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,adam=True,reg=True)
grad5,grad5_up = l5.backprop(grad6+decay_dilated_rate*(grad8_Input + grad8 + grad7),learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True,reg=True)
grad5_a,grad5_a_up = l5_a.backprop(grad5+decay_dilated_rate*(grad8_Input + grad8 + grad7 + grad6),learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,adam=True)

grad4_Input = tf_repeat(grad5_a,[1,2,2,1])
grad4,grad4_up = l4.backprop(grad4_Input,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True)
grad3,grad3_up = l3.backprop(grad4+decay_dilated_rate*(grad4_Input),learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,adam=True,reg=True)
grad2,grad2_up = l2.backprop(grad3+decay_dilated_rate*(grad4_Input+grad4),learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True,reg=True,stride=2)
grad1,grad1_up = l1.backprop(grad2,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,adam=True,reg=True)
grad1_a,grad1_a_up = l1_a.backprop(grad1+decay_dilated_rate*(grad2),learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True)

grad0,grad0_up = l0.backprop(grad1_a,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,awsgrad=True,reg=True)

grad_update = grad13_up + \
              grad12_up + grad11_up + grad10_up + grad9_up + grad9_a_up + \
              grad8_up + grad7_up + grad6_up + grad5_up + grad5_a_up + \
              grad4_up + grad3_up + grad2_up + grad1_up + grad1_a_up + \
              grad0_up               

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]
    data_input_type = 0 

    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)

        for batch_size_index in range(0,len(train_batch),batch_size//2):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size//2]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size//2]

            # data aug
            seq = iaa.Sequential([  
                iaa.Sometimes( (0.3),
                    iaa.CropAndPad(
                        percent=(0, 0.2),
                        pad_mode=["constant", "edge"],
                        pad_cval=(0, 128)
                    )
                ),
                iaa.Fliplr(1.0), # Horizontal flips
            ], random_order=True) # apply augmenters in random order

            images_aug = seq.augment_images(current_batch.astype(np.float32))
            current_batch = np.vstack((current_batch,images_aug)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            
            input_sess_array = [cost,accuracy,correct_prediction,grad_update]
            input_feed_dict= {x:current_batch,y:current_batch_label,
            iter_variable:iter,learning_rate_dynamic:learning_rate,batch_size_dynamic:batch_size}

            # online data augmentation here and standard normalization
            current_batch[:,:,:,0]  = (current_batch[:,:,:,0] - current_batch[:,:,:,0].mean(axis=0)) / ( current_batch[:,:,:,0].std(axis=0)+ 1e-20)
            current_batch[:,:,:,1]  = (current_batch[:,:,:,1] - current_batch[:,:,:,1].mean(axis=0)) / ( current_batch[:,:,:,1].std(axis=0)+ 1e-20)
            current_batch[:,:,:,2]  = (current_batch[:,:,:,2] - current_batch[:,:,:,2].mean(axis=0)) / ( current_batch[:,:,:,2].std(axis=0)+ 1e-20)
            current_batch,current_batch_label  = shuffle(current_batch,current_batch_label)
            # online data augmentation here and standard normalization

            sess_result = sess.run(input_sess_array,feed_dict=input_feed_dict  )
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],
            ' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]
            sess_result = sess.run([cost,accuracy,correct_prediction],
            feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,learning_rate_dynamic:learning_rate,batch_size_dynamic:batch_size})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],
            ' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n---------- Learning Rate : ", learning_rate * (1.0/(1.0+learnind_rate_decay*iter))," Data: ",data_input_type )
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size//2)),' Current Acc: ', 
            train_acca/(len(train_batch)/(batch_size//2) ),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', 
            test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/batch_size//2))
        train_cot.append(train_cota/(len(train_batch)/batch_size//2))
        test_acc.append(test_acca/(len(test_batch)/batch_size))
        test_cot.append(test_cota/(len(test_batch)/batch_size))
        test_cota,test_acca = 0,0
        train_cota,train_acca = 0,0
        # data_input_type = data_input_type + 1
        # if data_input_type == 4: data_input_type = 0 

    # Normalize the cost of the training
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))
    test_cot = (test_cot-min(test_cot) ) / (max(test_cot)-min(test_cot))

    # training done now plot
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig("Case a Train.png")

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("Case a Test.png")





# -- end code --