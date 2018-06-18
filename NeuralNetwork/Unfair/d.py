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
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32)  +  \
(tf_elu(tf.cast(tf.less_equal(x,0),tf.float32) * x)+1.0)

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
# data aug
seq = iaa.Sequential([
    iaa.Fliplr(1.0), # Horizonatl flips
], random_order=True) # apply augmenters in random order
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
   
# ================= LAYER CLASSES =================

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
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

# simple normalizaiton
train_batch  = train_batch/255.0
test_batch  = test_batch/255.0

# hyper parameter
num_epoch = 21
batch_size = 50
print_size = 1

learning_rate = 0.0002
learnind_rate_decay = 0.0
beta1,beta2,adam_e = 0.9,0.999,1e-8

# define class here
channel_sizes = 128
l1 = CNN(3,3,channel_sizes,stddev=0.05)
l2 = CNN(3,channel_sizes,channel_sizes,stddev=0.04)
l3 = CNN(3,channel_sizes,channel_sizes,stddev=0.06)

l4 = CNN(3,channel_sizes,channel_sizes,stddev=0.04)
l5 = CNN(3,channel_sizes,channel_sizes,stddev=0.06)
l6 = CNN(3,channel_sizes,channel_sizes,stddev=0.05)

l7 = CNN(3,channel_sizes,channel_sizes,stddev=0.04)
l8 = CNN(1,channel_sizes,channel_sizes,stddev=0.05)
l9 = CNN(1,channel_sizes,10,stddev=0.06)

all_weights = [
    l1.getw(),l2.getw(),l3.getw(),
    l4.getw(),l5.getw(),l6.getw(),
    l7.getw(),l8.getw(),l9.getw()
    ]

# graph
x = tf.placeholder(shape=[batch_size,32,32,3],dtype=tf.float32)
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

final_global = tf.reduce_mean(layer9,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y) )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def unfair_grad_1(): 
    '''
        Def: Peform unfair back propagation on block 1

        Return:
            Array of Update Tensors
    '''

    # u1
    grad_prepare_u1 = tf_repeat(tf.reshape(final_soft-y,[batch_size,1,1,10]),[1,8,8,1])

    grad9_u1,grad9_up_u1 = l9.backprop_unfair(grad_prepare_u1,learning_rate_change=learning_rate_change) 

    layer9_u1 = l9.feedforward(layer8) 

    final_global_u1 = tf.reduce_mean(layer9_u1,[1,2])
    final_soft_u1 = tf_softmax(final_global_u1)

    # u2 
    grad_prepare_u2 = tf_repeat(tf.reshape(final_soft_u1-y,[batch_size,1,1,10]),[1,8,8,1])

    grad9_u2,grad9_up_u2 = l9.backprop_unfair(grad_prepare_u2,learning_rate_change=learning_rate_change) 
    grad8_u2,grad8_up_u2 = l8.backprop_unfair(grad9_u2,learning_rate_change=learning_rate_change) 

    layer8_u2 = l8.feedforward(layer7) 
    layer9_u2 = l9.feedforward(layer8_u2) 

    final_global_u2 = tf.reduce_mean(layer9_u2,[1,2])
    final_soft_u2 = tf_softmax(final_global_u2)

    # u3 
    grad_prepare_u3 = tf_repeat(tf.reshape(final_soft_u2-y,[batch_size,1,1,10]),[1,8,8,1])

    grad9_u3,grad9_up_u3 = l9.backprop_unfair(grad_prepare_u3,learning_rate_change=learning_rate_change) 
    grad8_u3,grad8_up_u3 = l8.backprop_unfair(grad9_u3,learning_rate_change=learning_rate_change) 
    grad7_u3,grad7_up_u3 = l7.backprop_unfair(grad8_u3,learning_rate_change=learning_rate_change) 

    layer7_u3 = l7.feedforward(layer7_Input) 
    layer8_u3 = l8.feedforward(layer7_u3) 
    layer9_u3 = l9.feedforward(layer8_u3) 

    final_global_u3 = tf.reduce_mean(layer9_u3,[1,2])
    final_soft_u3 = tf_softmax(final_global_u3)

    # u4 
    grad_prepare_u4 = tf_repeat(tf.reshape(final_soft_u3-y,[batch_size,1,1,10]),[1,8,8,1])

    grad9_u4,grad9_up_u4 = l9.backprop_unfair(grad_prepare_u4,learning_rate_change=learning_rate_change) 
    grad8_u4,grad8_up_u4 = l8.backprop_unfair(grad9_u4,learning_rate_change=learning_rate_change) 
    grad7_u4,grad7_up_u4 = l7.backprop_unfair(grad8_u4,learning_rate_change=learning_rate_change) 

    grad6_Input_u4 = tf_repeat(grad7_u4,[1,2,2,1])
    grad6_u4,grad6_up_u4 = l6.backprop_unfair(grad6_Input_u4,learning_rate_change=learning_rate_change) 

    layer6_u4 = l6.feedforward(layer5) 
    layer7_Input_u4 = tf.nn.avg_pool(layer6_u4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    layer7_u4 = l7.feedforward(layer7_Input_u4) 
    layer8_u4 = l8.feedforward(layer7_u4) 
    layer9_u4 = l9.feedforward(layer8_u4) 

    final_global_u4 = tf.reduce_mean(layer9_u4,[1,2])
    final_soft_u4 = tf_softmax(final_global_u4)

    # u5
    grad_prepare_u5 = tf_repeat(tf.reshape(final_soft_u4-y,[batch_size,1,1,10]),[1,8,8,1])

    grad9_u5,grad9_up_u5 = l9.backprop_unfair(grad_prepare_u5,learning_rate_change=learning_rate_change) 
    grad8_u5,grad8_up_u5 = l8.backprop_unfair(grad9_u5,learning_rate_change=learning_rate_change) 
    grad7_u5,grad7_up_u5 = l7.backprop_unfair(grad8_u5,learning_rate_change=learning_rate_change) 

    grad6_Input_u5 = tf_repeat(grad7_u5,[1,2,2,1])
    grad6_u5,grad6_up_u5 = l6.backprop_unfair(grad6_Input_u5,learning_rate_change=learning_rate_change) 
    grad5_u5,grad5_up_u5 = l5.backprop_unfair(grad6_u5,learning_rate_change=learning_rate_change) 

    layer5_u5 = l5.feedforward(layer4) 
    layer6_u5 = l6.feedforward(layer5_u5) 
    layer7_Input_u5 = tf.nn.avg_pool(layer6_u5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    layer7_u5 = l7.feedforward(layer7_Input_u5) 
    layer8_u5 = l8.feedforward(layer7_u5) 
    layer9_u5 = l9.feedforward(layer8_u5) 

    final_global_u5 = tf.reduce_mean(layer9_u5,[1,2])
    final_soft_u5 = tf_softmax(final_global_u5)

    # u6
    grad_prepare_u6 = tf_repeat(tf.reshape(final_soft_u5-y,[batch_size,1,1,10]),[1,8,8,1])

    grad9_u6,grad9_up_u6 = l9.backprop_unfair(grad_prepare_u6,learning_rate_change=learning_rate_change) 
    grad8_u6,grad8_up_u6 = l8.backprop_unfair(grad9_u6,learning_rate_change=learning_rate_change) 
    grad7_u6,grad7_up_u6 = l7.backprop_unfair(grad8_u6,learning_rate_change=learning_rate_change) 

    grad6_Input_u6 = tf_repeat(grad7_u6,[1,2,2,1])
    grad6_u6,grad6_up_u6 = l6.backprop_unfair(grad6_Input_u6,learning_rate_change=learning_rate_change) 
    grad5_u6,grad5_up_u6 = l5.backprop_unfair(grad6_u6,learning_rate_change=learning_rate_change) 
    grad4_u6,grad4_up_u6 = l4.backprop_unfair(grad5_u6,learning_rate_change=learning_rate_change) 

    layer4_u6 = l4.feedforward(layer4_Input) 
    layer5_u6 = l5.feedforward(layer4_u6) 
    layer6_u6 = l6.feedforward(layer5_u6) 
    layer7_Input_u6 = tf.nn.avg_pool(layer6_u6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    layer7_u6 = l7.feedforward(layer7_Input_u6) 
    layer8_u6 = l8.feedforward(layer7_u6) 
    layer9_u6 = l9.feedforward(layer8_u6) 

    final_global_u6 = tf.reduce_mean(layer9_u6,[1,2])
    final_soft_u6 = tf_softmax(final_global_u6)

    # u7
    grad_prepare_u7 = tf_repeat(tf.reshape(final_soft_u6-y,[batch_size,1,1,10]),[1,8,8,1])

    grad9_u7,grad9_up_u7 = l9.backprop_unfair(grad_prepare_u7,learning_rate_change=learning_rate_change) 
    grad8_u7,grad8_up_u7 = l8.backprop_unfair(grad9_u7,learning_rate_change=learning_rate_change) 
    grad7_u7,grad7_up_u7 = l7.backprop_unfair(grad8_u7,learning_rate_change=learning_rate_change) 

    grad6_Input_u7 = tf_repeat(grad7_u7,[1,2,2,1])
    grad6_u7,grad6_up_u7 = l6.backprop_unfair(grad6_Input_u7,learning_rate_change=learning_rate_change) 
    grad5_u7,grad5_up_u7 = l5.backprop_unfair(grad6_u7,learning_rate_change=learning_rate_change) 
    grad4_u7,grad4_up_u7 = l4.backprop_unfair(grad5_u7,learning_rate_change=learning_rate_change) 

    grad3_Input_u7 = tf_repeat(grad4_u7,[1,2,2,1])
    grad3_u7,grad3_up_u7 = l3.backprop_unfair(grad3_Input_u7,learning_rate_change=learning_rate_change) 

    layer3_u7 = l3.feedforward(layer2) 

    layer4_Input_u7 = tf.nn.avg_pool(layer3_u7,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    layer4_u7 = l4.feedforward(layer4_Input_u7) 
    layer5_u7 = l5.feedforward(layer4_u7) 
    layer6_u7 = l6.feedforward(layer5_u7) 

    layer7_Input_u7 = tf.nn.avg_pool(layer6_u7,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    layer7_u7 = l7.feedforward(layer7_Input_u7) 
    layer8_u7 = l8.feedforward(layer7_u7) 
    layer9_u7 = l9.feedforward(layer8_u7) 

    final_global_u7 = tf.reduce_mean(layer9_u7,[1,2])
    final_soft_u7 = tf_softmax(final_global_u7)

    # u8
    grad_prepare_u8 = tf_repeat(tf.reshape(final_soft_u7-y,[batch_size,1,1,10]),[1,8,8,1])

    grad9_u8,grad9_up_u8 = l9.backprop_unfair(grad_prepare_u8,learning_rate_change=learning_rate_change) 
    grad8_u8,grad8_up_u8 = l8.backprop_unfair(grad9_u8,learning_rate_change=learning_rate_change) 
    grad7_u8,grad7_up_u8 = l7.backprop_unfair(grad8_u8,learning_rate_change=learning_rate_change) 

    grad6_Input_u8 = tf_repeat(grad7_u8,[1,2,2,1])
    grad6_u8,grad6_up_u8 = l6.backprop_unfair(grad6_Input_u8,learning_rate_change=learning_rate_change) 
    grad5_u8,grad5_up_u8 = l5.backprop_unfair(grad6_u8,learning_rate_change=learning_rate_change) 
    grad4_u8,grad4_up_u8 = l4.backprop_unfair(grad5_u8,learning_rate_change=learning_rate_change) 

    grad3_Input_u8 = tf_repeat(grad4_u8,[1,2,2,1])
    grad3_u8,grad3_up_u8 = l3.backprop_unfair(grad3_Input_u8,learning_rate_change=learning_rate_change) 
    grad2_u8,grad2_up_u8 = l2.backprop_unfair(grad3_u8,learning_rate_change=learning_rate_change) 

    layer2_u8 = l2.feedforward(layer1) 
    layer3_u8 = l3.feedforward(layer2_u8) 

    layer4_Input_u8 = tf.nn.avg_pool(layer3_u8,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    layer4_u8 = l4.feedforward(layer4_Input_u8) 
    layer5_u8 = l5.feedforward(layer4_u8) 
    layer6_u8 = l6.feedforward(layer5_u8) 

    layer7_Input_u8 = tf.nn.avg_pool(layer6_u8,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    layer7_u8 = l7.feedforward(layer7_Input_u8) 
    layer8_u8 = l8.feedforward(layer7_u8) 
    layer9_u8 = l9.feedforward(layer8_u8) 

    final_global_u8 = tf.reduce_mean(layer9_u8,[1,2])
    final_soft_u8 = tf_softmax(final_global_u8)

    # u9
    grad_prepare_u9 = tf_repeat(tf.reshape(final_soft_u8-y,[batch_size,1,1,10]),[1,8,8,1])

    grad9_u9,grad9_up_u9 = l9.backprop_unfair(grad_prepare_u9,learning_rate_change=learning_rate_change) 
    grad8_u9,grad8_up_u9 = l8.backprop_unfair(grad9_u9,learning_rate_change=learning_rate_change) 
    grad7_u9,grad7_up_u9 = l7.backprop_unfair(grad8_u9,learning_rate_change=learning_rate_change) 

    grad6_Input_u9 = tf_repeat(grad7_u9,[1,2,2,1])
    grad6_u9,grad6_up_u9 = l6.backprop_unfair(grad6_Input_u9,learning_rate_change=learning_rate_change) 
    grad5_u9,grad5_up_u9 = l5.backprop_unfair(grad6_u9,learning_rate_change=learning_rate_change) 
    grad4_u9,grad4_up_u9 = l4.backprop_unfair(grad5_u9,learning_rate_change=learning_rate_change) 

    grad3_Input_u9 = tf_repeat(grad4_u9,[1,2,2,1])
    grad3_u9,grad3_up_u9 = l3.backprop_unfair(grad3_Input_u9,learning_rate_change=learning_rate_change) 
    grad2_u9,grad2_up_u9 = l2.backprop_unfair(grad3_u9,learning_rate_change=learning_rate_change) 
    grad1_u9,grad1_up_u9 = l1.backprop_unfair(grad2_u9,learning_rate_change=learning_rate_change) 

    layer1_u9 = l1.feedforward(x) 
    layer2_u9 = l2.feedforward(layer1_u9) 
    layer3_u9 = l3.feedforward(layer2_u9) 

    layer4_Input_u9 = tf.nn.avg_pool(layer3_u9,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    layer4_u9 = l4.feedforward(layer4_Input_u9) 
    layer5_u9 = l5.feedforward(layer4_u9) 
    layer6_u9 = l6.feedforward(layer5_u9) 

    layer7_Input_u9 = tf.nn.avg_pool(layer6_u9,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    layer7_u9 = l7.feedforward(layer7_Input_u9) 
    layer8_u9 = l8.feedforward(layer7_u9) 
    layer9_u9 = l9.feedforward(layer8_u9) 

    final_global_u9 = tf.reduce_mean(layer9_u9,[1,2])
    final_soft_u9 = tf_softmax(final_global_u9)

    # back prop again but this time fully
    grad_prepare = tf_repeat(tf.reshape(final_soft_u9-y,[batch_size,1,1,10]),[1,8,8,1])
    grad9,grad9_up = l9.backprop_unfair(grad_prepare,learning_rate_change=learning_rate_change) 
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
                grad9_up_u1 + \
                grad9_up_u2 + grad8_up_u2 + \
                grad9_up_u3 + grad8_up_u3 + grad7_up_u3 + \
                grad9_up_u4 + grad8_up_u4 + grad7_up_u4 + grad6_up_u4 + \
                grad9_up_u5 + grad8_up_u5 + grad7_up_u5 + grad6_up_u5 + grad5_up_u5 + \
                grad9_up_u6 + grad8_up_u6 + grad7_up_u6 + grad6_up_u6 + grad5_up_u6 + grad4_up_u6 + \
                grad9_up_u7 + grad8_up_u7 + grad7_up_u7 + grad6_up_u7 + grad5_up_u7 + grad4_up_u7 + grad3_up_u7 + \
                grad9_up_u8 + grad8_up_u8 + grad7_up_u8 + grad6_up_u8 + grad5_up_u8 + grad4_up_u8 + grad3_up_u8 + grad2_up_u8 + \
                grad9_up_u9 + grad8_up_u9 + grad7_up_u9 + grad6_up_u9 + grad5_up_u9 + grad4_up_u9 + grad3_up_u9 + grad2_up_u9 + grad1_up_u9 + \
                grad9_up + grad8_up + grad7_up + \
                grad6_up + grad5_up + grad4_up + \
                grad3_up + grad2_up + grad1_up 

    return grad_update

# choose a random unfair grad
grad_update_1 = unfair_grad_1()

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
        which_grad,which_grad_print = grad_update_1,1

        for batch_size_index in range(0,len(train_batch),batch_size//2):
            
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size//2]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size//2]

            # online data augmentation here and standard normalization
            images_aug  = seq.augment_images(current_batch.astype(np.float32))
            current_batch = np.vstack((current_batch,images_aug)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch,current_batch_label  = shuffle(current_batch,current_batch_label)
            # online data augmentation here and standard normalization

            sess_result = sess.run([cost,accuracy,correct_prediction,which_grad],
            feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,learning_rate_dynamic:learning_rate,phase:True})
            print("Current Iter : ",iter," Using grad: ",which_grad_print ," current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
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
            print(" Using grad: ",which_grad_print,'Train Current cost: ', train_cota/(len(train_batch)/(batch_size//2)),' Current Acc: ', train_acca/(len(train_batch)/(batch_size//2) ),end='\n')
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