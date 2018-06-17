import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread,imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from imgaug import augmenters as iaa
import imgaug as ia

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(678)
tf.set_random_seed(678)
ia.seed(678)

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

# ================= VIZ =================
def show_hist_of_weigt(all_weight_list,status='before'):
    fig = plt.figure()
    weight_index = 0

    for i in range(1,1+int(len(all_weight_list)//3)):
        ax = fig.add_subplot(1,4,i)
        ax.grid(False)
        temp_weight_list = all_weight_list[weight_index:weight_index+3]
        for temp_index in range(len(temp_weight_list)):
            current_flat = temp_weight_list[temp_index].flatten()
            ax.hist(current_flat,histtype='step',bins='auto',label=str(temp_index+weight_index))
            ax.legend()
        ax.set_title('From Layer : '+str(weight_index+1)+' to '+str(weight_index+3))
        weight_index = weight_index + 3
    plt.savefig('viz/weights_'+str(status)+"_training.png")
    plt.close('all')

# Def: Simple function to show 9 image with different channels
def show_9_images(image,layer_num,image_num,channel_increase=3,alpha=None,gt=None,predict=None):
    image = (image-image.min())/(image.max()-image.min())
    fig = plt.figure()
    color_channel = 0
    limit = 10
    if alpha: limit = len(gt)
    for i in range(1,limit):
        ax = fig.add_subplot(3,3,i)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if alpha:
            ax.set_title("GT: "+str(gt[i-1])+" Predict: "+str(predict[i-1]))
        else:
            ax.set_title("Channel : " + str(color_channel) + " : " + str(color_channel+channel_increase-1))
        ax.imshow(np.squeeze(image[:,:,color_channel:color_channel+channel_increase]))
        color_channel = color_channel + channel_increase
    
    if alpha:
        plt.savefig('viz/z_'+str(alpha) + "_alpha_image.png")
    else:
        plt.savefig('viz/'+str(layer_num) + "_layer_"+str(image_num)+"_image.png")
    plt.close('all')
# ================= VIZ =================


# ================= DATA AUGMENTATION =================
seq = iaa.Sequential([
    iaa.Fliplr(1.0), 
], random_order=True) 
seq1 = iaa.Sequential([
    iaa.Flipud(1.0), 
], random_order=True) 
seq2 = iaa.Sequential([
    iaa.Fliplr(1.0), 
    iaa.Flipud(1.0), 
], random_order=True) 
# ================= DATA AUGMENTATION =================



# ================= LAYER CLASSES =================
class CNN():
    
    def __init__(self,k,inc,out,stddev):
        self.w = tf.Variable(tf.truncated_normal([k,k,inc,out],stddev=stddev))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding) 
        self.layerA = tf_elu(self.layer)
        return self.layerA 

    def backprop_fair(self,gradient,learning_rate_change,stride=1,padding='SAME'):
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
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v_prev / (1-beta2)
        adam_middel = learning_rate_change/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat)  )))         

        # === DUMMY UPDATES =====
        update_w.append(tf.assign( self.w,self.w ))
        update_w.append(tf.assign( self.m,self.m ))
        update_w.append(tf.assign( self.v_prev,self.v_prev ))
        # === DUMMY UPDATES =====
          
        return grad_pass,update_w 

    def backprop_unfair(self,gradient,learning_rate_change,stride=1,padding='SAME'):
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
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v_prev / (1-beta2)
        adam_middel = learning_rate_change/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat)  )))         

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
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v_prev / (1-beta2)
        adam_middel = learning_rate_change/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat)  )))   

        return grad_pass,update_w   
# ================= LAYER CLASSES =================



# stl 10 data
PathDicom = "../../Dataset/STL10/stl10_binary/"

train_file   = open(PathDicom+"train_X.bin",'rb')
train_label_f  = open(PathDicom+"train_y.bin",'rb')

test_file   = open(PathDicom+"test_X.bin",'rb')
test_label_f  = open(PathDicom+"test_y.bin",'rb')

# Train set have 500 images per class 5000
# Test set have 800 images per class 8000
# So I am going to switch 
onehot_encoder = OneHotEncoder(sparse=True)

x_data = np.fromfile(train_file, dtype=np.uint8)
x_data = np.reshape(x_data, (-1, 3, 96, 96))
x_data = np.transpose(x_data, (0, 3, 2, 1))
train_label_pure = np.expand_dims(np.fromfile(train_label_f, dtype=np.uint8).astype(np.float64),axis=1)
train_label = onehot_encoder.fit_transform(train_label_pure).toarray().astype(np.float32)

y_data = np.fromfile(test_file, dtype=np.uint8)
y_data = np.reshape(y_data, (-1, 3, 96, 96))
y_data = np.transpose(y_data, (0, 3, 2, 1))
test_label = np.expand_dims(np.fromfile(test_label_f, dtype=np.uint8).astype(np.float64),axis=1)
test_label = onehot_encoder.fit_transform(test_label).toarray().astype(np.float32)

train_batch = np.zeros((len(x_data),64,64,3))
test_batch = np.zeros((len(y_data),64,64,3))

for x in range(len(x_data)):
    train_batch[x,:,:,:] = imresize(x_data[x,:,:,:],(64,64))
for x in range(len(y_data)):
    test_batch[x,:,:,:] =  imresize(y_data[x,:,:,:],(64,64))

train_batch = train_batch[:40,:,:,:]
train_label = train_label[:40,:]
test_batch = test_batch[:40,:,:,:]
test_label = test_label[:40,:]

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

# simple normalize
train_batch = train_batch/255.0
test_batch = test_batch/255.0

# Number of each classes
# 1 -> airplane
# 2 -> bird
# 4 -> cat
# 5 -> deer
# 6 -> Dog
# 7 -> horse
# 8 -> monkey
# 9 -> ship
# 10 -> truck

# hyper parameter
num_epoch = 21
batch_size = 8
print_size = 1

learning_rate = 0.000008
learnind_rate_decay = 0.0
beta1,beta2,adam_e = 0.9,0.999,1e-8

# define class here
channel_sizes = 256
l1 = CNN(3,3,channel_sizes,stddev=0.04)
l2 = CNN(3,channel_sizes,channel_sizes,stddev=0.05)
l3 = CNN(3,channel_sizes,channel_sizes,stddev=0.06)

l4 = CNN(3,channel_sizes,channel_sizes,stddev=0.04)
l5 = CNN(3,channel_sizes,channel_sizes,stddev=0.05)
l6 = CNN(2,channel_sizes,channel_sizes,stddev=0.06)

l7 = CNN(3,channel_sizes,channel_sizes,stddev=0.06)
l8 = CNN(3,channel_sizes,channel_sizes,stddev=0.05)
l9 = CNN(3,channel_sizes,channel_sizes,stddev=0.04)

l10 = CNN(3,channel_sizes,channel_sizes,stddev=0.06)
l11 = CNN(1,channel_sizes,channel_sizes,stddev=0.05)
l12 = CNN(1,channel_sizes,10,stddev=0.04)

all_weights = [
    l1.getw(),l2.getw(),l3.getw(),
    l4.getw(),l5.getw(),l6.getw(),
    l7.getw(),l8.getw(),l9.getw(),
    l10.getw(),l11.getw(),l12.getw()
    ]

# graph
x = tf.placeholder(shape=[batch_size,64,64,3],dtype=tf.float32)
y = tf.placeholder(shape=[batch_size,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_dynamic  = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate_dynamic * (1.0/(1.0+learnind_rate_decay*iter_variable))
phase = tf.placeholder(tf.bool)

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1) 
layer3 = l3.feedforward(layer2)

layer4_Input = tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input)
layer5 = l5.feedforward(layer4) 
layer6 = l6.feedforward(layer5) 

layer7_Input = tf.nn.avg_pool(layer6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer7 = l7.feedforward(layer7_Input)
layer8 = l8.feedforward(layer7) 
layer9 = l9.feedforward(layer8) 

layer10_Input = tf.nn.avg_pool(layer9,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer10 = l10.feedforward(layer10_Input)
layer11 = l11.feedforward(layer10) 
layer12 = l12.feedforward(layer11) 

final_global = tf.reduce_mean(layer12,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y) )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def unfair_grad(): 
    '''
        Def: Peform unfair back propagation

        Return:
            Array of Update Tensors
    '''

    # back prop via the first block
    grad_prepare_u1 = tf_repeat(tf.reshape(final_soft-y,[batch_size,1,1,10]),[1,1,1,1])
    grad12_u1,grad12_up_u1 = l12.backprop_unfair(grad_prepare_u1,learning_rate_change=learning_rate_change)
    grad11_u1,grad11_up_u1 = l11.backprop_unfair(grad12_u1,learning_rate_change=learning_rate_change) 
    grad10_u1,grad10_up_u1 = l10.backprop_unfair(grad11_u1,learning_rate_change=learning_rate_change) 

    # perform feed forward operation and prepare for again back prop
    layer10_u1 = l10.feedforward(layer10_Input)
    layer11_u1 = l11.feedforward(layer10_u1) 
    layer12_u1 = l12.feedforward(layer11_u1)  
    final_global_u1 = tf.reduce_mean(layer12_u1,[1,2])
    final_soft_u1 = tf_softmax(final_global_u1)

    grad_prepare = tf_repeat(tf.reshape(final_soft_u1-y,[batch_size,1,1,10]),[1,1,1,1])
    grad12,grad12_up = l12.backprop_unfair(grad_prepare,learning_rate_change=learning_rate_change)
    grad11,grad11_up = l11.backprop_unfair(grad12,learning_rate_change=learning_rate_change) 
    grad10,grad10_up = l10.backprop_unfair(grad11,learning_rate_change=learning_rate_change) 

    grad9_Input = tf_repeat(grad10,[1,2,2,1])
    grad9,grad9_up = l9.backprop_unfair(grad9_Input,learning_rate_change=learning_rate_change) 
    grad8,grad8_up = l8.backprop_unfair(grad9,learning_rate_change=learning_rate_change) 
    grad7,grad7_up = l7.backprop_unfair(grad8,learning_rate_change=learning_rate_change) 

    grad6_Input = tf_repeat(grad7,[1,2,2,1])
    grad6,grad6_up = l6.backprop_unfair(grad6_Input,learning_rate_change=learning_rate_change)
    grad5,grad5_up = l5.backprop_unfair(grad6,learning_rate_change=learning_rate_change)
    grad4,grad4_up = l4.backprop_unfair(grad5,learning_rate_change=learning_rate_change)

    grad3_Input = tf_repeat(grad4,[1,2,2,1])
    grad3,grad3_up = l3.backprop_unfair(grad3_Input,learning_rate_change=learning_rate_change)
    grad2,grad2_up = l2.backprop_unfair(grad3,learning_rate_change=learning_rate_change)
    grad1,grad1_up = l1.backprop_unfair(grad2,learning_rate_change=learning_rate_change)

    grad_update =\
                grad12_up_u1 + grad11_up_u1 + grad10_up_u1 + \
                grad12_up + grad11_up + grad10_up + \
                grad9_up + grad8_up + grad7_up + \
                grad6_up + grad5_up + grad4_up + \
                grad3_up + grad2_up + grad1_up + s

    return grad_update

def fair_grad(): 
    '''
        Def: Peform fair back propagation

        Return:
            Array of Update Tensors with additional self dummy updates
    '''
    grad_prepare = tf_repeat(tf.reshape(final_soft-y,[batch_size,1,1,10]),[1,1,1,1])
    grad12,grad12_up = l12.backprop_fair(grad_prepare,learning_rate_change=learning_rate_change)
    grad11,grad11_up = l11.backprop_fair(grad12,learning_rate_change=learning_rate_change) 
    grad10,grad10_up = l10.backprop_fair(grad11,learning_rate_change=learning_rate_change) 

    grad9_Input = tf_repeat(grad10,[1,2,2,1])
    grad9,grad9_up = l9.backprop_unfair(grad9_Input,learning_rate_change=learning_rate_change) 
    grad8,grad8_up = l8.backprop_unfair(grad9,learning_rate_change=learning_rate_change) 
    grad7,grad7_up = l7.backprop_unfair(grad8,learning_rate_change=learning_rate_change) 

    grad6_Input = tf_repeat(grad7,[1,2,2,1])
    grad6,grad6_up = l6.backprop_unfair(grad6_Input,learning_rate_change=learning_rate_change)
    grad5,grad5_up = l5.backprop_unfair(grad6,learning_rate_change=learning_rate_change)
    grad4,grad4_up = l4.backprop_unfair(grad5,learning_rate_change=learning_rate_change)

    grad3_Input = tf_repeat(grad4,[1,2,2,1])
    grad3,grad3_up = l3.backprop_unfair(grad3_Input,learning_rate_change=learning_rate_change)
    grad2,grad2_up = l2.backprop_unfair(grad3,learning_rate_change=learning_rate_change)
    grad1,grad1_up = l1.backprop_unfair(grad2,learning_rate_change=learning_rate_change)

    grad_update = \
                grad12_up + grad11_up + grad10_up + \
                grad9_up + grad8_up + grad7_up + \
                grad6_up + grad5_up + grad4_up + \
                grad3_up + grad2_up + grad1_up 
    return grad_update

# Every 2 iteration we are going to perform unfair back prop
grad_update = tf.cond( tf.equal(tf.mod(iter_variable,2),0.0),
        true_fn= unfair_grad,
        false_fn=fair_grad
        )


# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    

    # ------- histogram of weights before training ------
    show_hist_of_weigt(sess.run(all_weights),status='before')
    # ------- histogram of weights before training ------

    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    # start the training
    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)

        for batch_size_index in range(0,len(train_batch),batch_size//4):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size//4]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size//4]

            # online data augmentation here and standard normalization
            images_aug  = seq.augment_images(current_batch.astype(np.float32))
            images_aug1 = seq.augment_images(current_batch.astype(np.float32))
            images_aug2 = seq.augment_images(current_batch.astype(np.float32))

            current_batch1 = np.vstack((current_batch,images_aug)).astype(np.float32)
            current_batch2 = np.vstack((images_aug1,images_aug2)).astype(np.float32)
            current_batch = np.vstack((current_batch1,current_batch2)).astype(np.float32)

            current_batch_label1 = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch_label2 = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label1,current_batch_label2)).astype(np.float32)

            current_batch,current_batch_label  = shuffle(current_batch,current_batch_label)
            # online data augmentation here and standard normalization

            sess_result = sess.run([cost,accuracy,correct_prediction,grad_update],
            feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,learning_rate_dynamic:learning_rate,phase:True})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]

        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size].astype(np.float32)
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size].astype(np.float32)
            sess_result = sess.run([cost,accuracy,correct_prediction],
            feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,phase:False})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],
            ' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n---------- Learning Rate : ", learning_rate * (1.0/(1.0+learnind_rate_decay*iter)) )
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size//4)),' Current Acc: ', train_acca/(len(train_batch)/(batch_size//4) ),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/(batch_size//4)))
        train_cot.append(train_cota/(len(train_batch)/(batch_size//4)))
        test_acc.append(test_acca/(len(test_batch)/batch_size))
        test_cot.append(test_cota/(len(test_batch)/batch_size))
        test_cota,test_acca = 0,0
        train_cota,train_acca = 0,0
    sys.exit()

    # Normalize the cost of the training
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))
    test_cot = (test_cot-min(test_cot) ) / (max(test_cot)-min(test_cot))

    # plot the training and testing graph
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig("viz/Case Train.png")

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("viz/Case Test.png")
    plt.close('all')

    # ------- histogram of weights after training ------
    show_hist_of_weigt(sess.run(all_weights),status='After')
    # ------- histogram of weights after training ------

    # get random 50 images from the test set and vis the gradient
    test_batch = test_batch[:batch_size,:,:,:]
    test_label = test_label[:batch_size,:]

    # ------ layer wise activation -------
    layer3_values = sess.run(layer3,feed_dict={x:test_batch})
    for immage_index in range(layer3_values.shape[0]):
        show_9_images(layer3_values[immage_index,:,:,:],3,immage_index)

    layer6_values = sess.run(layer6,feed_dict={x:test_batch})
    for immage_index in range(layer6_values.shape[0]):
        show_9_images(layer6_values[immage_index,:,:,:],6,immage_index)

    layer9_values = sess.run(layer9,feed_dict={x:test_batch})
    for immage_index in range(layer9_values.shape[0]):
        show_9_images(layer9_values[immage_index,:,:,:],9,immage_index)

    layer12_values = sess.run(layer12,feed_dict={x:test_batch})
    for immage_index in range(layer12_values.shape[0]):
        show_9_images(layer12_values[immage_index,:,:,:],12,immage_index,channel_increase=1)
    # ------ layer wise activation -------

    # -------- Interior Gradients -----------
    final_prediction_argmax = None
    final_gt_argmax = None
    for alpha_values in [0.02,0.04,0.06,0.08,0.1,0.3,0.6,0.8,1.0]:

        test_batch_a = (test_batch * alpha_values).astype(np.float32)
        sess_result = sess.run([cost,accuracy,correct_prediction,final_soft,grad1],feed_dict={x:test_batch_a,y:test_label,iter_variable:1.0,phase:False})
        
        final_prediction_argmax = [-1] + list(np.argmax(sess_result[3],axis=1))
        final_gt_argmax         = [-1] + list(np.argmax(test_label,axis=1))

        grad_important = sess_result[4]
        grad_important = np.sum(grad_important,axis=3)  
        grad_important = (grad_important-grad_important.min())/(grad_important.max()-grad_important.min())
        test_batch = (test_batch-test_batch.min())/(test_batch.max()-test_batch.min())
        overlayed_image = np.repeat(np.expand_dims(grad_important,axis=3),3,axis=3) * test_batch
        stacked_images = overlayed_image[0,:,:,:]
        for stacking in range(0,len(overlayed_image)):
            stacked_images = np.vstack((stacked_images.T,overlayed_image[stacking,:,:,:].T)).T

        show_9_images(stacked_images,0,0,alpha=alpha_values,gt=final_gt_argmax,predict=final_prediction_argmax)
    # -------- Interior Gradients -----------

    # -------- Intergral Gradients ----------
    base_line = test_batch * 0.0
    difference = test_batch - base_line
    step_size = 1000

    running_example = test_batch * 0.0
    for rim in range(1,step_size+1):
        current_alpha = rim / step_size
        test_batch_a = current_alpha * test_batch
        sess_result = sess.run([cost,accuracy,correct_prediction,final_soft,grad1],feed_dict={x:test_batch_a,y:test_label,iter_variable:1.0,phase:False})
        running_example = running_example + sess_result[4]
        final_prediction_argmax = sess_result[3]

    running_example = running_example * difference
    running_example = np.sum(running_example,axis=3)  
    running_example = (running_example-running_example.min())/(running_example.max()-running_example.min())
    overlayed_image = np.repeat(np.expand_dims(running_example,axis=3),3,axis=3) * test_batch
    stacked_images = overlayed_image[0,:,:,:]
    for stacking in range(0,len(overlayed_image)):
        stacked_images = np.vstack((stacked_images.T,overlayed_image[stacking,:,:,:].T)).T

    final_prediction_argmax = [-1] + list(np.argmax(final_prediction_argmax,axis=1))
    show_9_images(stacked_images,0,0,alpha=-99,gt=final_gt_argmax,predict=final_prediction_argmax)
    # -------- Intergral Gradients ----------


# -- end code --