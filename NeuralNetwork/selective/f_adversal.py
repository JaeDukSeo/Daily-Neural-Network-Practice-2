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

# class
class CNN():
    
    def __init__(self,k,inc,out,act,d_act):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))
        self.act,self.d_act = act,d_act
    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME',droprate=1.0):
        self.input  = input
        self.layer  = tf.nn.dropout(tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding) ,droprate)
        self.layerA = self.act(self.layer)
        return self.layerA 

    def backprop(self,gradient,learning_rate_change,stride=1,padding='SAME',mom=False,adam=False,awsgrad=False,reg=False):
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
            input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
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
            if reg: adam_middel = adam_middel - learning_rate_change * 0.0001 * self.w
            update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel  )) )    

        if awsgrad:
            update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   ))
            v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 
            def f1(): return v_t
            def f2(): return self.v_hat_prev
            v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
            adam_middel = tf.multiply(learning_rate_change/(tf.sqrt(v_max) + adam_e),self.m)
            if reg: adam_middel = adam_middel - learning_rate_change * 0.0001 * self.w
            update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel  )  ))
            update_w.append(tf.assign( self.v_prev,v_t ))
            update_w.append(tf.assign( self.v_hat_prev,v_max ))        

        if mom:
            update_w.append(tf.assign( self.m,self.m*beta1 + learning_rate* (grad)   ))
            adam_middel = self.m
            if reg: adam_middel = adam_middel - learning_rate_change * 0.0001 * self.w
            update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel )) )    

        return grad_pass,update_w  
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

learning_rate = 0.00025
learnind_rate_decay = 0.01

proportion_rate = 0.00001
decay_rate = 1

# define class
channel_size = 164
l1 = CNN(3,3,channel_size,tf_elu,d_tf_elu)
l2 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l3 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)

l4 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l5 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l6 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)

l7 = CNN(3,channel_size,channel_size,tf_elu,d_tf_elu)
l8 = CNN(1,channel_size,channel_size,tf_elu,d_tf_elu)
l9 = CNN(1,channel_size,10,tf_elu,d_tf_elu)

# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_dynamic  = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate_dynamic * (1.0/(1.0+learnind_rate_decay*iter_variable))
decay_dilated_rate   = proportion_rate       

droprate1 = tf.placeholder(tf.float32, shape=())
droprate2 = tf.placeholder(tf.float32, shape=())
droprate3 = tf.placeholder(tf.float32, shape=())

layer1 = l1.feedforward(x,droprate=droprate1)
layer2 = l2.feedforward(layer1,droprate=droprate2)
layer3 = l3.feedforward(layer2,stride=2,droprate=droprate3)

layer4 = l4.feedforward(layer3,droprate=droprate2)
layer5 = l5.feedforward(layer4,droprate=droprate3)
layer6 = l6.feedforward(layer5+decay_dilated_rate*layer4,stride=2,droprate=droprate1)

layer7 = l7.feedforward(layer6,droprate=droprate3)
layer8 = l8.feedforward(layer7,padding='VALID',droprate=droprate1)
layer9 = l9.feedforward(layer8+decay_dilated_rate*layer7,padding='VALID',droprate=droprate2)

final_global = tf.reduce_mean(layer9,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y) )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ===== auto train =====
# auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate_change,momentum=0.9).minimize(cost)
# auto_train = tf.train.RMSPropOptimizer(learning_rate=learning_rate_change).minimize(cost)
# auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate_change).minimize(cost)
# ===== auto train =====

# ===== manual ====
grad_prepare = tf.reshape(final_soft-y,[batch_size,1,1,10])
grad9,grad9_up = l9.backprop(grad_prepare,learning_rate_change=learning_rate_change,padding='VALID',awsgrad=True)
grad8,grad8_up = l8.backprop(grad9,learning_rate_change=learning_rate_change,padding='VALID',adam=True,reg=True)
grad7,grad7_up = l7.backprop(grad8+decay_dilated_rate*grad9,learning_rate_change=learning_rate_change,awsgrad=True)

grad6,grad6_up = l6.backprop(grad7,learning_rate_change=learning_rate_change,stride=2,adam=True)
grad5,grad5_up = l5.backprop(grad6,learning_rate_change=learning_rate_change,awsgrad=True,reg=True)
grad4,grad4_up = l4.backprop(grad5+decay_dilated_rate*grad6,learning_rate_change=learning_rate_change,adam=True)

grad3,grad3_up = l3.backprop(grad4,learning_rate_change=learning_rate_change,stride=2,awsgrad=True)
grad2,grad2_up = l2.backprop(grad3,learning_rate_change=learning_rate_change,adam=True,reg=True)
grad1,grad1_up = l1.backprop(grad2+decay_dilated_rate*grad3,learning_rate_change=learning_rate_change,awsgrad=True)

grad_update = grad9_up + grad8_up+ grad7_up + grad6_up + grad5_up + grad4_up + grad3_up + grad2_up + grad1_up
# ===== manual ====

# ====== adversal input ==========
adversal_x = x + 0.08 * tf.sign(grad1)

layer1_ad = l1.feedforward(adversal_x,droprate=droprate1)
layer2_ad = l2.feedforward(layer1_ad,droprate=droprate2)
layer3_ad = l3.feedforward(layer2_ad,stride=2,droprate=droprate3)

layer4_ad = l4.feedforward(layer3_ad,droprate=droprate2)
layer5_ad = l5.feedforward(layer4_ad,droprate=droprate3)
layer6_ad = l6.feedforward(layer5_ad+decay_dilated_rate*layer4_ad,stride=2,droprate=droprate1)

layer7_ad = l7.feedforward(layer6_ad,droprate=droprate3)
layer8_ad = l8.feedforward(layer7_ad,padding='VALID',droprate=droprate1)
layer9_ad = l9.feedforward(layer8_ad+decay_dilated_rate*layer7_ad,padding='VALID',droprate=droprate2)

final_global_ad = tf.reduce_mean(layer9_ad,[1,2])
final_soft_ad = tf_softmax(final_global_ad)

cost_ad = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_soft_ad,labels=y) )
correct_prediction_ad = tf.equal(tf.argmax(final_soft_ad, 1), tf.argmax(y, 1))
accuracy_ad = tf.reduce_mean(tf.cast(correct_prediction_ad, tf.float32))
# ====== adversal input ==========

# ===== adversal manual ====
grad_prepare_ad = tf.reshape(final_soft_ad-y,[batch_size,1,1,10])
grad9_ad,grad9_up_ad = l9.backprop(grad_prepare_ad,learning_rate_change=learning_rate_change,padding='VALID',awsgrad=True)
grad8_ad,grad8_up_ad = l8.backprop(grad9_ad,learning_rate_change=learning_rate_change,padding='VALID',adam=True,reg=True)
grad7_ad,grad7_up_ad = l7.backprop(grad8_ad+decay_dilated_rate*grad9_ad,learning_rate_change=learning_rate_change,awsgrad=True)

grad6_ad,grad6_up_ad = l6.backprop(grad7_ad,learning_rate_change=learning_rate_change,stride=2,adam=True)
grad5_ad,grad5_up_ad = l5.backprop(grad6_ad,learning_rate_change=learning_rate_change,awsgrad=True,reg=True)
grad4_ad,grad4_up_ad = l4.backprop(grad5_ad+decay_dilated_rate*grad6_ad,learning_rate_change=learning_rate_change,adam=True)

grad3_ad,grad3_up_ad = l3.backprop(grad4_ad,learning_rate_change=learning_rate_change,stride=2,awsgrad=True)
grad2_ad,grad2_up_ad = l2.backprop(grad3_ad,learning_rate_change=learning_rate_change,adam=True,reg=True)
grad1_ad,grad1_up_ad = l1.backprop(grad2_ad+decay_dilated_rate*grad3_ad,learning_rate_change=learning_rate_change,awsgrad=True)

grad_update = grad9_up_ad + grad8_up_ad+ grad7_up_ad + grad6_up_ad + grad5_up_ad + grad4_up_ad + grad3_up_ad + grad2_up_ad + grad1_up_ad
# ===== adversal manual ====


# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)

        lower_bound = 0.5 * (iter+1)/num_epoch
        random_drop1 = np.random.uniform(low=0.9+lower_bound,high=1.0)
        random_drop2 = np.random.uniform(low=0.9+lower_bound,high=1.0)
        random_drop3 = np.random.uniform(low=0.9+lower_bound,high=1.0)

        for batch_size_index in range(0,len(train_batch),batch_size//2):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size//2]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size//2]

            # data aug
            seq = iaa.Sequential([  
                iaa.Sometimes(0.1 + lower_bound ,
                    iaa.Affine(
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    )
                ),
                iaa.Sometimes(0.15 + lower_bound,
                    iaa.Affine(
                        rotate=(-25, 25),
                    )
                ),
                iaa.Sometimes(0.1 + lower_bound,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    )
                ),
                iaa.Fliplr(1.0), # Horizontal flips
            ], random_order=True) # apply augmenters in random order

            # online data augmentation here and standard normalization
            images_aug = seq.augment_images(current_batch.astype(np.float32))
            current_batch = np.vstack((current_batch,images_aug)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch[:,:,:,0]  = (current_batch[:,:,:,0] - current_batch[:,:,:,0].mean(axis=0)) / ( current_batch[:,:,:,0].std(axis=0))
            current_batch[:,:,:,1]  = (current_batch[:,:,:,1] - current_batch[:,:,:,1].mean(axis=0)) / ( current_batch[:,:,:,1].std(axis=0))
            current_batch[:,:,:,2]  = (current_batch[:,:,:,2] - current_batch[:,:,:,2].mean(axis=0)) / ( current_batch[:,:,:,2].std(axis=0))
            current_batch,current_batch_label  = shuffle(current_batch,current_batch_label)
            # online data augmentation here and standard normalization

            if iter % 2 == 0:
                sess_result = sess.run([cost,accuracy,correct_prediction,grad_update],feed_dict={x:current_batch,y:current_batch_label,
                iter_variable:iter,learning_rate_dynamic:learning_rate,droprate1:1.0,droprate2:1.0,droprate3:1.0})
            else:
                sess_result = sess.run([cost_ad,accuracy_ad,correct_prediction_ad,grad_update_ad,adversal_x],feed_dict={x:current_batch,y:current_batch_label,
                iter_variable:iter,learning_rate_dynamic:learning_rate,droprate1:1.0,droprate2:1.0,droprate3:1.0})
                
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],
            ' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]
            sess_result = sess.run([cost,accuracy,correct_prediction],
            feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,learning_rate_dynamic:learning_rate,
            droprate1:1.0,droprate2:1.0,droprate3:1.0})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],
            ' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n---------- Learning Rate : ", learning_rate * (1.0/(1.0+learnind_rate_decay*iter)) )
            print("Lower Bound : ",lower_bound,' Random  Lower: ',lower_bound ,
            '\n',"Drop 1 : ",random_drop1," Drop 2: ",random_drop2," Drop 3: ",random_drop3)
            
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size//2)),' Current Acc: ', 
            train_acca/(len(train_batch)/(batch_size//2) ),end='\n')

            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', 
            test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/batch_size))
        train_cot.append(train_cota/(len(train_batch)/batch_size))
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
    plt.savefig("Case a Train.png")

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("Case a Test.png")





# -- end code --