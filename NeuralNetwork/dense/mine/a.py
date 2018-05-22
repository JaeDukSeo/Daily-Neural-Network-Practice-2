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
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.truncated_normal([k,k,inc,out],stddev=0.05))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,phase,stride=1,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding)
        self.layer =  tf.layers.batch_normalization(self.layer, center=True, scale=True, training=phase)
        self.layerA = tf_elu(self.layer)
        return  self.layerA 

    def backprop(self,gradient,learning_rate_change,stride=1,padding='SAME',amsgrad=False,adam=False,mom=False,reg=False):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_elu(self.layer) 
        grad_part_3 = self.input
        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,strides=[1,stride,stride,1],padding=padding
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size*2] + list(grad_part_3.shape[1:]),
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
            update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,self.m))))
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
            if reg: adam_middel = adam_middel - learning_rate_change * decouple_weigth * self.w
            update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel  )) ) 

        elif mom:
            # === MoM ====
            update_w.append(tf.assign( self.m, self.m*0.9 +  learning_rate_change * mom_plus  * (grad)   ))
            adam_middel = self.m
            if reg: adam_middel = adam_middel - learning_rate_change * decouple_weigth * self.w
            update_w.append(tf.assign( self.w, tf.subtract(self.w,adam_middel)  ))
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

    def backprop(self,gradient):
        return 1

# # data
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

test_batch = test_batch/255.0
train_batch = train_batch/255.0

# Normalize the image range from 0 to 1 and then standardization
# test_batch[:,:,:,0]  = (test_batch[:,:,:,0] - test_batch[:,:,:,0].min(axis=0)) / ( test_batch[:,:,:,0].max(axis=0) -test_batch[:,:,:,0].min(axis=0) )
# test_batch[:,:,:,1]  = (test_batch[:,:,:,1] - test_batch[:,:,:,1].min(axis=0)) / ( test_batch[:,:,:,1].max(axis=0)-test_batch[:,:,:,1].min(axis=0) )
# test_batch[:,:,:,2]  = (test_batch[:,:,:,2] - test_batch[:,:,:,2].min(axis=0)) / ( test_batch[:,:,:,2].max(axis=0) -test_batch[:,:,:,2].min(axis=0) )

# test_batch[:,:,:,0]  = (test_batch[:,:,:,0] - test_batch[:,:,:,0].mean(axis=0)) / ( test_batch[:,:,:,0].std(axis=0) + 1e-10)
# test_batch[:,:,:,1]  = (test_batch[:,:,:,1] - test_batch[:,:,:,1].mean(axis=0)) / ( test_batch[:,:,:,1].std(axis=0) + 1e-10)
# test_batch[:,:,:,2]  = (test_batch[:,:,:,2] - test_batch[:,:,:,2].mean(axis=0)) / ( test_batch[:,:,:,2].std(axis=0) + 1e-10)

# train_batch[:,:,:,0]  = (train_batch[:,:,:,0] - train_batch[:,:,:,0].min(axis=0)) / ( train_batch[:,:,:,0].max(axis=0)-train_batch[:,:,:,0].min(axis=0) )
# train_batch[:,:,:,1]  = (train_batch[:,:,:,1] - train_batch[:,:,:,1].min(axis=0)) / ( train_batch[:,:,:,1].max(axis=0)-train_batch[:,:,:,1].min(axis=0) )
# train_batch[:,:,:,2]  = (train_batch[:,:,:,2] - train_batch[:,:,:,2].min(axis=0)) / ( train_batch[:,:,:,2].max(axis=0) -train_batch[:,:,:,2].min(axis=0) )

# train_batch[:,:,:,0]  = (train_batch[:,:,:,0] - train_batch[:,:,:,0].mean(axis=0)) / ( train_batch[:,:,:,0].std(axis=0) + 1e-10)
# train_batch[:,:,:,1]  = (train_batch[:,:,:,1] - train_batch[:,:,:,1].mean(axis=0)) / ( train_batch[:,:,:,1].std(axis=0) + 1e-10)
# train_batch[:,:,:,2]  = (train_batch[:,:,:,2] - train_batch[:,:,:,2].mean(axis=0)) / ( train_batch[:,:,:,2].std(axis=0) + 1e-10)

# hyper
num_epoch = 51
batch_size = 50
print_size = 1
beta1,beta2,adam_e = 0.9,0.9,1e-8
decouple_weigth = 0.00001

mom_plus = 0.0001
learning_rate = 0.0003
learning_rate_decay = 0.0

proportion_rate = 0.009
decay_rate = 0.0

channel_size = 64

# 32 * 32 the first
l0 = CNN(3,3,channel_size)

# 32 * 32
l1 = CNN(3,channel_size,channel_size)
l2 = CNN(3,channel_size*2,channel_size*2)
l3 = CNN(3,channel_size*4,channel_size)

# 16 * 16
l4 = CNN(3,channel_size,channel_size)
l5 = CNN(3,channel_size*2,channel_size*2)
l6 = CNN(3,channel_size*4,channel_size)

# 8 * 8
l7 = CNN(3,channel_size,channel_size)
l8 = CNN(3,channel_size*2,channel_size*2)
l9 = CNN(3,channel_size*4,channel_size)

# 4 * 4
l10 = CNN(3,channel_size,channel_size)
l11 = CNN(3,channel_size*2,channel_size*2)
l12 = CNN(3,channel_size*4,channel_size)

# 2 * 2
l13 = CNN(3,channel_size,channel_size)
l14 = CNN(3,channel_size*2,channel_size*2)
l15 = CNN(3,channel_size*4,channel_size)

# 1 * 1
l16 = CNN(1,channel_size,channel_size)
l17 = CNN(1,channel_size*2,channel_size*2)
l18 = CNN(1,channel_size*4,10)


# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate * (1.0/(1.0+learning_rate_decay*iter_variable))
decay_dilated_rate = proportion_rate  * (1.0/(1.0+decay_rate*iter_variable))
phase = tf.placeholder(tf.bool)

layer0 = l0.feedforward(x,phase=phase)

layer1 = l1.feedforward(layer0,phase=phase)
layer2 = l2.feedforward(tf.concat([layer1,layer0],axis=3),phase=phase)
layer3 = l3.feedforward(tf.concat([layer2,layer1,layer0],axis=3),phase=phase)
# layer3 = tf.nn.dropout(layer3,0.5)

layer4_Input = tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
# layer4_Input =  tf.layers.batch_normalization(layer4_Input, center=True, scale=True, training=phase)
layer4 = l4.feedforward(layer4_Input,phase=phase)
layer5 = l5.feedforward(tf.concat([layer4,layer4_Input],axis=3),phase=phase)
layer6 = l6.feedforward(tf.concat([layer5,layer4,layer4_Input],axis=3),phase=phase)
# layer6 = tf.nn.dropout(layer6,0.5)

layer7_Input = tf.nn.avg_pool(layer6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
# layer7_Input =  tf.layers.batch_normalization(layer7_Input, center=True, scale=True, training=phase)
layer7 = l7.feedforward(layer7_Input,phase=phase)
layer8 = l8.feedforward(tf.concat([layer7,layer7_Input],axis=3),phase=phase)
layer9 = l9.feedforward(tf.concat([layer8,layer7,layer7_Input],axis=3),phase=phase)
# layer9 = tf.nn.dropout(layer9,0.5)

layer10_Input = tf.nn.avg_pool(layer9,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
# layer10_Input =  tf.layers.batch_normalization(layer10_Input, center=True, scale=True, training=phase)
layer10 = l10.feedforward(layer10_Input,phase=phase)
layer11 = l11.feedforward(tf.concat([layer10,layer10_Input],axis=3),phase=phase)
layer12 = l12.feedforward(tf.concat([layer11,layer10,layer10_Input],axis=3),phase=phase)
# layer12 = tf.nn.dropout(layer12,0.5)

layer13_Input = tf.nn.avg_pool(layer12,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
# layer13_Input =  tf.layers.batch_normalization(layer13_Input, center=True, scale=True, training=phase)
layer13 = l13.feedforward(layer13_Input,phase=phase)
layer14 = l14.feedforward(tf.concat([layer13,layer13_Input],axis=3),phase=phase)
layer15 = l15.feedforward(tf.concat([layer14,layer13,layer13_Input],axis=3),phase=phase)
# layer15 = tf.nn.dropout(layer15,0.5)

layer16_Input = tf.nn.avg_pool(layer15,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
# layer16_Input =  tf.layers.batch_normalization(layer16_Input, center=True, scale=True, training=phase)
layer16 = l16.feedforward(layer16_Input,phase=phase)
layer17 = l17.feedforward(tf.concat([layer16,layer16_Input],axis=3),phase=phase)
layer18 = l18.feedforward(tf.concat([layer17,layer16,layer16_Input],axis=3),phase=phase)

final_global = tf.reduce_mean(layer18,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate_change,beta2=0.9).minimize(cost)

# # ==== Manual Back Prop ======
# grad_prepare = tf.reshape(final_soft-y,[batch_size*2,1,1,10])
# grad9,grad9_up = l9.backprop(grad_prepare,learning_rate_change=learning_rate_change,padding='VALID',mom=True)
# grad8,grad8_up = l8.backprop(grad9,learning_rate_change=learning_rate_change,padding='VALID',amsgrad=True)
# grad7,grad7_up = l7.backprop(grad8+decay_dilated_rate*(grad9),learning_rate_change=learning_rate_change,adam=True,reg=False)

# grad6_Input = tf_repeat(grad7,[1,2,2,1])
# grad6,grad6_up = l6.backprop(grad6_Input,learning_rate_change=learning_rate_change,mom=True)
# grad5,grad5_up = l5.backprop(grad6+decay_dilated_rate*(grad6_Input),learning_rate_change=learning_rate_change,amsgrad=True)
# grad4,grad4_up = l4.backprop(grad5+decay_dilated_rate*(grad6+grad6_Input),learning_rate_change=learning_rate_change,adam=True,reg=False)

# grad3_Input = tf_repeat(grad4,[1,2,2,1])
# grad3,grad3_up = l3.backprop(grad3_Input,learning_rate_change=learning_rate_change,mom=True)
# grad2,grad2_up = l2.backprop(grad3+decay_dilated_rate*(grad3_Input),learning_rate_change=learning_rate_change,amsgrad=True)
# grad1,grad1_up = l1.backprop(grad2+decay_dilated_rate*(grad3+grad3_Input),learning_rate_change=learning_rate_change,adam=True,reg=False)

# grad_update = grad9_up + grad8_up+ grad7_up + grad6_up + grad5_up + grad4_up + grad3_up + grad2_up + grad1_up
# # ==== Manual Back Prop ======

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):


        # Shuffle the training data every time changing the std value and the mean everytime
        train_batch,train_label = shuffle(train_batch,train_label)
        
        for batch_size_index in range(0,len(train_batch),batch_size//2):
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

            sess_result = sess.run([cost,accuracy,auto_train,extra_update_ops],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,phase:True})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]
            current_batch[:,:,:,0]  = (current_batch[:,:,:,0] - current_batch[:,:,:,0].mean(axis=0)) / ( current_batch[:,:,:,0].std(axis=0) + 1e-10)
            current_batch[:,:,:,1]  = (current_batch[:,:,:,1] - current_batch[:,:,:,1].mean(axis=0)) / ( current_batch[:,:,:,1].std(axis=0) + 1e-10)
            current_batch[:,:,:,2]  = (current_batch[:,:,:,2] - current_batch[:,:,:,2].mean(axis=0)) / ( current_batch[:,:,:,2].std(axis=0) + 1e-10)
            sess_result = sess.run([cost,accuracy,correct_prediction],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,phase:False})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n---------- LR : ", learning_rate * (1.0/(1.0+learning_rate_decay*iter)) )
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
    plt.savefig("Case Train.png")

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("Case Test.png")





# -- end code --