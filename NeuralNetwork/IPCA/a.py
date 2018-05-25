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
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32)  + ( tf_elu(tf.cast(tf.less_equal(x,0),tf.float32) * x) + 1.0)
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

# data aug
seq = iaa.Sequential([
    iaa.Sometimes(0.1,
        iaa.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        )
    ),
    iaa.Sometimes(0.1,
        iaa.Affine(
            rotate=(-25, 25),
        )
    ),
    iaa.Sometimes(0.1,
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        )
    ),
    iaa.Fliplr(1.0), # Horizonatl flips
], random_order=True) # apply augmenters in random order

# class
class CNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))
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
        update_w.append(
            tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   )
        )
        v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 

        def f1(): return v_t
        def f2(): return self.v_hat_prev

        v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
        adam_middel = tf.multiply(learning_rate_change/(tf.sqrt(v_max) + adam_e),self.m)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel  )  ))
        update_w.append(tf.assign( self.v_prev,v_t ))
        update_w.append(tf.assign( self.v_hat_prev,v_max ))        
        return grad_pass,update_w   

# Followed the implmentation from: https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
# Followed the implmentation from: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
class batch_norm():
    
    def __init__(self,dim,channel):
        
        self.gamma = tf.Variable(tf.ones(shape=[dim,dim,channel]))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.gamma)),tf.Variable(tf.zeros_like(self.gamma))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.gamma))

        # for one update
        self.input = None
        self.current_mean = None
        self.current_var = None
        self.x_norm = None

        # exp moving average
        self.moving_mean = tf.Variable(tf.zeros(shape=[dim,dim,channel]))
        self.moving_var  = tf.Variable(tf.zeros(shape=[dim,dim,channel]))

    def feedforward(self,input,is_training):
        moving_update = []

        self.input = input
        self.current_mean,self.current_var = tf.nn.moments(input,axes=0)

        def training_fn(): 
            # Update the moving average
            self.x_norm = (input - self.current_mean) / (tf.sqrt(self.current_var + 1e-8))
            moving_update.append(tf.assign(self.moving_mean,self.moving_mean*0.9 + self.current_mean*0.1 ))
            moving_update.append(tf.assign(self.moving_var,self.moving_var*0.9 + self.current_var*0.1 ))
            return self.x_norm,moving_update

        def testing_fn(): 
            # In the Testing Data use the moving average  
            self.x_norm = (input-self.moving_mean)/ (tf.sqrt(self.moving_var + 1e-8))
            return self.x_norm ,moving_update

        self.x_norm,moving_update = tf.cond(is_training, true_fn=training_fn, false_fn=testing_fn)
        self.out = self.gamma * self.x_norm
        return self.out,moving_update

    def backprop(self,gradient):
        
        grad_mean_prep = self.input - self.current_mean
        grad_var_prep  = 1. / tf.sqrt(self.current_var + 1e-8)

        grad_norm = gradient * self.gamma
        grad_var  = tf.reduce_sum(grad_norm * grad_mean_prep, axis=0) * -.5 * grad_var_prep ** 3
        grad_mean = tf.reduce_sum(grad_norm * -1.0 * grad_var_prep, axis=0) + grad_var * tf.reduce_mean(-2. * grad_mean_prep, axis=0)
        
        grad_pass = (grad_norm * grad_var_prep) + (grad_var * 2 * grad_mean_prep / batch_size) + (grad_mean / batch_size    )
        grad = tf.reduce_sum(gradient * self.x_norm , axis=0)

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   ))
        v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 

        def f1(): return v_t
        def f2(): return self.v_hat_prev
        v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
        adam_middel = tf.multiply(learning_rate_change/(tf.sqrt(v_max) + adam_e),self.m)
        update_w.append(tf.assign(self.gamma,tf.subtract(self.gamma,adam_middel  )  ))
        update_w.append(tf.assign( self.v_prev,v_t ))
        update_w.append(tf.assign( self.v_hat_prev,v_max ))        
        return grad_pass,update_w   


class PCA_Layer():
    
    def __init__(self):
        print(3)

    def feedforward(self,input,is_training):
        print(78)

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

learning_rate = 0.0003
learnind_rate_decay = 0.001

beta1,beta2,adam_e = 0.9,0.9,1e-8
proportion_rate = 0.8
decay_rate = 10 

# define class
l1 = CNN(3,3,96)
l2 = CNN(3,96,96)
l3 = CNN(3,96,192)

b1 = batch_norm(32,192)
l4 = CNN(3,192,192)
l5 = CNN(3,192,192)
l6 = CNN(3,192,192)

b2 = batch_norm(16,192)
l7 = CNN(3,192,192)
l8 = CNN(1,192,192)
l9 = CNN(1,192,10)


# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_dynamic  = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate_dynamic * (1.0/(1.0+learnind_rate_decay*iter_variable))
decay_dilated_rate = proportion_rate / (1 + decay_rate * iter_variable)

phase = tf.placeholder(tf.bool)

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)

layer4_BN,layer4_BN_UP = b1.feedforward(layer3,is_training=phase)
layer4_Input = tf.nn.avg_pool(layer4_BN,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input)
layer5 = l5.feedforward(layer4)
layer6 = l6.feedforward(layer5)

layer7_BN,layer7_BN_U = b2.feedforward(layer6,is_training=phase)
layer7_Input = tf.nn.avg_pool(layer7_BN,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer7 = l7.feedforward(layer7_Input)
layer8 = l8.feedforward(layer7,padding='VALID')
layer9 = l9.feedforward(layer8,padding='VALID')

final_global = tf.reduce_mean(layer9,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y) )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate_change,beta2=0.9).minimize(cost)


# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)

        for batch_size_index in range(0,len(train_batch),batch_size//2):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size//2]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size//2]

            # online data augmentation here and standard normalization
            images_aug1 = seq.augment_images(current_batch.astype(np.float32))
            current_batch = np.vstack((current_batch,images_aug1)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch[:,:,:,0]  = (current_batch[:,:,:,0] - current_batch[:,:,:,0].mean(axis=0)) / ( current_batch[:,:,:,0].std(axis=0))
            current_batch[:,:,:,1]  = (current_batch[:,:,:,1] - current_batch[:,:,:,1].mean(axis=0)) / ( current_batch[:,:,:,1].std(axis=0))
            current_batch[:,:,:,2]  = (current_batch[:,:,:,2] - current_batch[:,:,:,2].mean(axis=0)) / ( current_batch[:,:,:,2].std(axis=0))
            current_batch,current_batch_label  = shuffle(current_batch,current_batch_label)
            # online data augmentation here and standard normalization
            sess_result = sess.run([cost,accuracy,correct_prediction,grad_update],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,
                learning_rate_dynamic:learning_rate,phase:True})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]
            sess_result = sess.run([cost,accuracy,correct_prediction],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,
            learning_rate_dynamic:learning_rate,phase:False})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],
            ' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n---------- Learning Rate : ", learning_rate * (1.0/(1.0+learnind_rate_decay*iter)) )
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size//2)),' Current Acc: ', train_acca/(len(train_batch)/(batch_size//2) ),end='\n')
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
    plt.savefig("Case c Train.png")

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("Case c Test.png")





# -- end code --