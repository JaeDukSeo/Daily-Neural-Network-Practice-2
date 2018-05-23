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
def tf_softmax(x): return tf.nn.softmax(x)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# import time
# time.sleep(60*60*2)

# data aug
seq = iaa.Sequential([
    iaa.Fliplr(1.0), # horizontal flips
    iaa.Affine(
        scale={"x": (0.9, 1.2), "y": (0.9, 1.2)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-25, 25),
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
    
    def __init__(self,k,inc,out,stddev=0.05):
        self.w = tf.Variable(tf.truncated_normal([k,k,inc,out],stddev=stddev))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding)
        self.layerA = tf_elu(self.layer)
        return  self.layerA 
        
    def backprop(self,gradient,learning_rate_change,stride=1,padding='SAME',amsgrad=True,adam=False,mom=False):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_elu(self.layer) 
        grad_part_3 = self.input
        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,strides=[1,stride,stride,1],padding=padding
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
            filter= self.w,out_backprop = grad_middle,strides=[1,stride,stride,1],padding=padding
        )

        update_w = []
        if amsgrad:
            
            # === AMSGrad =====
            update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   ))
            v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 
            def f1(): return v_t
            def f2(): return self.v_hat_prev
            v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
            adam_middel = learning_rate_change/(tf.sqrt(v_max) + adam_e)
            if reg: adam_middel = adam_middel - learning_rate_change * decouple_weigth * self.w
            update_w.append(tf.assign( self.v_prev,v_t ))
            update_w.append(tf.assign( self.v_hat_prev,v_max ))        

        elif adam: 
            # === ADAM =====
            update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
            update_w.append(tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   ))
            m_hat = self.m / (1-beta1)
            v_hat = self.v_prev / (1-beta2)
            adam_middel = learning_rate_change/(tf.sqrt(v_hat) + adam_e)
            adam_middel = tf.multiply(adam_middel,m_hat)
            update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel  )) ) 

        elif mom:
            # === MoM ====
            update_w.append(tf.assign( self.m, self.m*0.9 +  learning_rate_change * mom_plus  * (grad)   ))
            adam_middel = self.m
            update_w.append(tf.assign( self.w, tf.subtract(self.w,adam_middel)  ))
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

# Normalize the image range from 0 to 1
test_batch,train_batch = test_batch/255.0,train_batch/255.0

# hyper
num_epoch = 31
batch_size = 50
print_size = 1

learning_rate = 0.0003
learning_rate_decay = 0.0

proportion_rate = 0.09
decay_rate = 0.0

mom_plus = 0.0001
beta1,beta2,adam_e = 0.9,0.9,1e-8

# define class
channel_size = 16
l0 = CNN(3,3,channel_size)

l1a = CNN(3,channel_size,channel_size,stddev=0.025)
l1b = CNN(3,channel_size,channel_size)
l2a = CNN(3,channel_size,channel_size,stddev=0.025)
l2b = CNN(3,channel_size,channel_size)
l3a = CNN(3,channel_size,channel_size*2)
l3b = CNN(3,channel_size,channel_size*2,stddev=0.025)

l4a = CNN(3,channel_size*2,channel_size*2)
l4b = CNN(3,channel_size*2,channel_size*2,stddev=0.025)
l5a = CNN(3,channel_size*2,channel_size*2)
l5b = CNN(3,channel_size*2,channel_size*2,stddev=0.025)
l6a = CNN(3,channel_size*2,channel_size*4)
l6b = CNN(3,channel_size*2,channel_size*4,stddev=0.025)

l7a = CNN(3,channel_size*4,channel_size*4,stddev=0.025)
l7b = CNN(3,channel_size*4,channel_size*4)
l8a = CNN(1,channel_size*4,channel_size*4,stddev=0.025)
l8b = CNN(1,channel_size*4,channel_size*4)
l9a = CNN(1,channel_size*4,channel_size*4,stddev=0.025)
l9b = CNN(1,channel_size*4,channel_size*4)

l10 = CNN(1,channel_size*4,10,stddev=0.025)

# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate * (1.0/(1.0+learning_rate_decay*iter_variable))
decay_dilated_rate = proportion_rate  * (1.0/(1.0+decay_rate*iter_variable))

shake_value = tf.placeholder(tf.float32, shape=[None,1,1,1])
shake_value_backprop = tf.placeholder(tf.float32, shape=[None,1,1,1])

layer0 = l0.feedforward(x)

layer1a = l1a.feedforward(layer0) * shake_value
layer1b = l1b.feedforward(layer0) * (1.0-shake_value)
layer2_Input = layer1a + layer1b 
layer2a = l2a.feedforward(layer2_Input) * shake_value
layer2b = l2bfeedforward(layer2_Input) * (1.0-shake_value)
layer3_Input = layer2a + layer2b 
layer3a = l3a.feedforward(layer3_Input) * shake_value
layer3b = l3b.feedforward(layer3_Input) * (1.0-shake_value)

layer4_Input = layer3a + layer3b 
layer4a = l4a.feedforward(layer4_Input,stride=2) * shake_value
layer4b = l4b.feedforward(layer4_Input,stride=2) * (1.0-shake_value)
layer5_Input = layer4a + layer4b
layer5a = l5a.feedforward(layer5_Input) * shake_value
layer5b = l5b.feedforward(layer5_Input) * (1.0-shake_value)
layer6_Input = layer5a + layer5b 
layer6a = l6a.feedforward(layer6_Input) * shake_value
layer6b = l6b.feedforward(layer6_Input) * (1.0-shake_value)

layer7_Input = layer6a + layer6b 
layer7a = l7a.feedforward(layer7_Input,stride=2) * shake_value
layer7b = l7b.feedforward(layer7_Input,stride=2) * (1.0-shake_value)
layer8_Input = layer7a + layer7b
layer8a = l8a.feedforward(layer8_Input,padding='VALID') * shake_value
layer8b = l8b.feedforward(layer8_Input,padding='VALID') * (1.0-shake_value)
layer9_Input = layer8a + layer8b 
layer9a = l9a.feedforward(layer9_Input,padding='VALID') * shake_value
layer9b = l9b.feedforward(layer9_Input,padding='VALID') * (1.0-shake_value)

layer10_Input = layer9a + layer9b 
layer10 = l10.feedforward(layer10_Input,padding='VALID')

final_global = tf.reduce_mean(layer10,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# === Manual Back ====
grad_prepare = tf.reshape(final_soft-y,[batch_size,1,1,10])
grad10,grad10_up = l10.backprop(grad_prepare,learning_rate_change=learning_rate_change,padding='VALID')

grad9a,grad9a_up = l9a.backprop(grad10,learning_rate_change=learning_rate_change,padding='VALID')* shake_value_backprop
grad9b,grad9b_up = l9b.backprop(grad10,learning_rate_change=learning_rate_change,padding='VALID')* (1.0-shake_value_backprop)
grad8_Input = grad9a + grad9b
grad8a,grad8a_up = l8a.backprop(grad8_Input+decay_dilated_rate*grad10,learning_rate_change=learning_rate_change,padding='VALID')* shake_value_backprop
grad8b,grad8b_up = l8b.backprop(grad8_Input+decay_dilated_rate*grad10,learning_rate_change=learning_rate_change,padding='VALID')* (1.0-shake_value_backprop)
grad7_Input = grad8a + grad8b
grad7a,grad7a_up = l7a.backprop(grad7_Input+decay_dilated_rate*(grad10+grad8_Input),learning_rate_change=learning_rate_change,stride=2)* shake_value_backprop
grad7b,grad7b_up = l7b.backprop(grad7_Input+decay_dilated_rate*(grad10+grad8_Input),learning_rate_change=learning_rate_change,stride=2)* (1.0-shake_value_backprop)

grad6_Input = grad7a + grad7b
grad6a,grad6a_up = l6a.backprop(grad6_Input,learning_rate_change=learning_rate_change)* shake_value_backprop
grad6b,grad6b_up = l6b.backprop(grad6_Input,learning_rate_change=learning_rate_change)* (1.0-shake_value_backprop)
grad5_Input = grad6a + grad6b
grad5a,grad5a_up = l5a.backprop(grad5_Input+decay_dilated_rate*grad6_Input,learning_rate_change=learning_rate_change)* shake_value_backprop
grad5b,grad5b_up = l5b.backprop(grad5_Input+decay_dilated_rate*grad6_Input,learning_rate_change=learning_rate_change)* (1.0-shake_value_backprop)
grad4_Input = grad5a + grad5b
grad4a,grad4a_up = l4a.backprop(grad4_Input+decay_dilated_rate*(grad6_Input+grad5_Input),learning_rate_change=learning_rate_change,stride=2)* shake_value_backprop
grad4b,grad4b_up = l4b.backprop(grad4_Input+decay_dilated_rate*(grad6_Input+grad5_Input),learning_rate_change=learning_rate_change,stride=2)* (1.0-shake_value_backprop)

grad3_Input = grad4a + grad4b
grad3a,grad3a_up = l3a.backprop(grad3_Input,learning_rate_change=learning_rate_change)* shake_value_backprop
grad3b,grad3b_up = l3b.backprop(grad3_Input,learning_rate_change=learning_rate_change)* (1.0-shake_value_backprop)
grad2_Input = grad3a + grad3b
grad2a,grad2a_up = l2a.backprop(grad2_Input+decay_dilated_rate*grad3_Input,learning_rate_change=learning_rate_change)* shake_value_backprop
grad2b,grad2b_up = l2b.backprop(grad2_Input+decay_dilated_rate*grad3_Input,learning_rate_change=learning_rate_change)* (1.0-shake_value_backprop)
grad1_Input = grad2a + grad2b
grad1a,grad1a_up = l1a.backprop(grad1_Input+decay_dilated_rate*(grad2_Input+grad3_Input),learning_rate_change=learning_rate_change)* shake_value_backprop
grad1b,grad1b_up = l1b.backprop(grad1_Input+decay_dilated_rate*(grad2_Input+grad3_Input),learning_rate_change=learning_rate_change)* (1.0-shake_value_backprop)

grad0_Input = grad1a + grad1b
grad0,grad0_up = l10.backprop(grad0_Input,learning_rate_change=learning_rate_change)

grad_update = grad10_up + \
              grad9a_up + grad9b_up + grad8a_up + grad8b_up + grad7a_up + grad7b_up + \
              grad4a_up + grad4b_up + grad5a_up + grad5b_up + grad6a_up + grad6b_up + \
              grad1a_up + grad1b_up + grad2a_up + grad2b_up + grad3a_up + grad3b_up + \
              grad0_up
# === Manual Back ====



# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)
        
        for batch_size_index in range(0,len(train_batch),(batch_size//2)):
            current_batch = train_batch[batch_size_index:batch_size_index+(batch_size//2)]
            current_batch_label = train_label[batch_size_index:batch_size_index+(batch_size//2)]

            # online data augmentation here and standard normalization
            images_aug = seq.augment_images(current_batch.astype(np.float32))
            current_batch = np.vstack((current_batch,images_aug)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch,current_batch_label = shuffle(current_batch,current_batch_label)
            current_batch[:,:,:,0]  = (current_batch[:,:,:,0] - current_batch[:,:,:,0].mean(axis=0)) / ( current_batch[:,:,:,0].std(axis=0) + 1e-10)
            current_batch[:,:,:,1]  = (current_batch[:,:,:,1] - current_batch[:,:,:,1].mean(axis=0)) / ( current_batch[:,:,:,1].std(axis=0) + 1e-10)
            current_batch[:,:,:,2]  = (current_batch[:,:,:,2] - current_batch[:,:,:,2].mean(axis=0)) / ( current_batch[:,:,:,2].std(axis=0) + 1e-10)
            # online data augmentation here and standard normalization

            sess_result = sess.run([cost,accuracy,grad_update],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,
            shake_value:np.random.uniform(low=0.0,high=1.0,size=(batch_size,1,1,1) ) ,shake_value_backprop:np.random.uniform(low=0.0,high=1.0,size=(batch_size,1,1,1) )  })
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]
            current_batch[:,:,:,0]  = (current_batch[:,:,:,0] - current_batch[:,:,:,0].mean(axis=0)) / ( current_batch[:,:,:,0].std(axis=0) + 1e-10)
            current_batch[:,:,:,1]  = (current_batch[:,:,:,1] - current_batch[:,:,:,1].mean(axis=0)) / ( current_batch[:,:,:,1].std(axis=0) + 1e-10)
            current_batch[:,:,:,2]  = (current_batch[:,:,:,2] - current_batch[:,:,:,2].mean(axis=0)) / ( current_batch[:,:,:,2].std(axis=0) + 1e-10)
            sess_result = sess.run([cost,accuracy,correct_prediction],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,shake_value:0.5})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n----------")
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size//2)),' Current Acc: ', train_acca/(len(train_batch)/(batch_size//2)),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/(batch_size//2)))
        train_cot.append(train_cota/(len(train_batch)/(batch_size//2)))
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
    plt.savefig("Case e Train.png")

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("Case e Test.png")





# -- end code --