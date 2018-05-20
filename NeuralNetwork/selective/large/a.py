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
def tf_softmax(x): return tf.nn.softmax(x)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Data Augmentation code from: http://imgaug.readthedocs.io/en/latest/source/examples_basics.html#a-standard-use-case
seq = iaa.Sequential([
    iaa.Fliplr(1.0), # horizontal flips
    iaa.Flipud(0.4), # horizontal flips
    iaa.Crop(percent=(0, 0.2)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.05))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 0.9)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*100), per_channel=0.2),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 0.9), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Sometimes(0.5,
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
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

    def backprop(self,gradient,learning_rate_change,batch_size_dynamic,stride=1,padding='SAME',adam=False,awsgrad=False,reg=False):
        grad_part_1 = gradient 
        grad_part_2 = self.d_act(self.layer) 
        grad_part_3 = self.input
        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,filter_sizes = self.w.shape,out_backprop = grad_middle,strides=[1,stride,stride,1],padding=padding
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size_dynamic] + list(grad_part_3.shape[1:]),filter= self.w,out_backprop = grad_middle,strides=[1,stride,stride,1],padding=padding
        )

        update_w = []

        if  adam:
            update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
            update_w.append(tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   ))
            m_hat = self.m / (1-beta1)
            v_hat = self.v_prev / (1-beta2)
            adam_middel = learning_rate_change/(tf.sqrt(v_hat) + adam_e)
            adam_middel = tf.multiply(adam_middel,m_hat)
            if reg: adam_middel = adam_middel - learning_rate_change * decouple_weight  * self.w
            update_w.append(tf.assign(self.w,tf.subtract(self.w,adam_middel  )) )    

        if awsgrad:
            update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   ))
            v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 
            def f1(): return v_t
            def f2(): return self.v_hat_prev
            v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
            adam_middel = tf.multiply(learning_rate_change/(tf.sqrt(v_max) + adam_e),self.m)
            if reg: adam_middel = adam_middel - learning_rate_change * decouple_weight  * self.w
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
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.int32)

print('--- Before Image Resize -----')
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

# Image Large with cv2 INTER_LANCZOS4
test_batch = np.asarray([cv2.resize(x.astype(np.float32),(126,126),interpolation=cv2.INTER_LANCZOS4) for x in test_batch ])

# standardize Normalize data per channel
test_batch[:,:,:,0]  = (test_batch[:,:,:,0] - test_batch[:,:,:,0].mean(axis=0)) / ( test_batch[:,:,:,0].std(axis=0)+ 1e-20)
test_batch[:,:,:,1]  = (test_batch[:,:,:,1] - test_batch[:,:,:,1].mean(axis=0)) / ( test_batch[:,:,:,1].std(axis=0)+ 1e-20)
test_batch[:,:,:,2]  = (test_batch[:,:,:,2] - test_batch[:,:,:,2].mean(axis=0)) / ( test_batch[:,:,:,2].std(axis=0)+ 1e-20)

print('--- Afer Image Resize (Only Test Image) -----')
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

# hyper parameter
num_epoch = 21 
batch_size =50
print_size = 1
beta1,beta2,adam_e = 0.9,0.9,1e-9
decouple_weight   = 0.000008

learning_rate = 0.0001
learnind_rate_decay = 0.001

proportion_rate = 0.0001
decay_rate = 0.01

# define layers

l1 = CNN(2,3,320,tf_elu,d_tf_elu)
l2 = CNN(2,320,320,tf_elu,d_tf_elu)
l3 = CNN(2,320,640,tf_elu,d_tf_elu)

l4 = CNN(2,640,640,tf_elu,d_tf_elu)
l5 = CNN(2,640,640,tf_elu,d_tf_elu)
l6 = CNN(2,640,960,tf_elu,d_tf_elu)

l7 = CNN(2,960,960,tf_elu,d_tf_elu)
l8 = CNN(2,960,960,tf_elu,d_tf_elu)
l9 = CNN(2,960,1280,tf_elu,d_tf_elu)

l10 = CNN(2,1280,1280,tf_elu,d_tf_elu)
l11 = CNN(2,1280,1280,tf_elu,d_tf_elu)
l12 = CNN(2,1280,1600,tf_elu,d_tf_elu)

l13 = CNN(2,1600,1600,tf_elu,d_tf_elu)
l14 = CNN(2,1600,1600,tf_elu,d_tf_elu)
l15 = CNN(2,1600,1920,tf_elu,d_tf_elu)

l16 = CNN(2,1920,1920,tf_elu,d_tf_elu)
l17 = CNN(1,1920,10,tf_elu,d_tf_elu)

# define graph
x = tf.placeholder(shape=[None,126,126,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

batch_size_dynamic= tf.placeholder(tf.int32, shape=())

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_dynamic  = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate_dynamic * (1.0/(1.0+learnind_rate_decay*iter_variable))
decay_dilated_rate   = proportion_rate       * (1.0/(1.0+decay_rate*iter_variable))

droprate1 = tf.placeholder(tf.float32, shape=())
droprate2 = tf.placeholder(tf.float32, shape=())
droprate3 = tf.placeholder(tf.float32, shape=())

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)

layer4_Input = tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
layer4 = l4.feedforward(layer4_Input)
layer5 = l5.feedforward(layer4)
layer6 = l6.feedforward(layer5)

layer7_Input = tf.nn.avg_pool(layer6,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
layer7 = l7.feedforward(layer7_Input)
layer8 = l8.feedforward(layer7)
layer9 = l9.feedforward(layer8)

layer10_Input = tf.nn.avg_pool(layer9,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
layer10 = l10.feedforward(layer10_Input)
layer111 = l11.feedforward(layer10)
layer112 = l12.feedforward(layer111)

layer13_Input = tf.nn.avg_pool(layer112,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
layer113 = l13.feedforward(layer13_Input)
layer114 = l14.feedforward(layer113)
layer115 = l15.feedforward(layer114)

layer16_Input = tf.nn.avg_pool(layer115,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
layer16 = l16.feedforward(layer16_Input)
layer17 = l17.feedforward(layer16,padding='VALID')

final_global = tf.reduce_mean(layer17,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y)  )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ===== manual ==== HAVE TO FINISH 
grad_prepare = tf.reshape(final_soft-y,[batch_size,1,1,10])

grad17,grad17_up = l17.backprop(grad_prepare,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic,padding='VALID',awsgrad=True,reg=True)
grad16,grad16_up = l16.backprop(grad17,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)

grad15_Input = tf_repeat(grad16,[1,2,2,1])
grad15,grad15_up = l15.backprop(grad15_Input,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad14,grad14_up = l14.backprop(grad15,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad13,grad13_up = l13.backprop(grad14,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)

grad12_Input = tf_repeat(grad13,[1,2,2,1])
grad12,grad12_up = l12.backprop(grad12_Input,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad11,grad11_up = l11.backprop(grad12,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad10,grad10_up = l10.backprop(grad11,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)

grad9_Input = tf_repeat(grad10,[1,2,2,1])
grad9,grad9_up = l9.backprop(grad9_Input,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad8,grad8_up = l8.backprop(grad9,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad7,grad7_up = l7.backprop(grad8,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)

grad6_Input = tf_repeat(grad7,[1,2,2,1])
grad6,grad6_up = l6.backprop(grad6_Input,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad5,grad5_up = l5.backprop(grad6,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad4,grad4_up = l4.backprop(grad5,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)

grad3_Input = tf_repeat(grad4,[1,2,2,1])
grad3,grad3_up = l3.backprop(grad3_Input,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad2,grad2_up = l2.backprop(grad3,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)
grad1,grad1_up = l1.backprop(grad2,learning_rate_change=learning_rate_change,batch_size_dynamic=batch_size_dynamic)

grad_update = grad17_up + grad16_up + \
              grad15_up + grad14_up + grad13_up  + \
              grad12_up + grad11_up + grad10_up  + \
              grad9_up + grad8_up + grad7_up  + \
              grad6_up + grad5_up + grad4_up  + \
              grad3_up + grad2_up + grad1_up  

# sess
with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)

        lower_bound = 0.015 * (iter+1)/num_epoch
        random_drop1 = np.random.uniform(low=0.985+lower_bound,high=1.000000000000001)
        random_drop2 = np.random.uniform(low=0.985+lower_bound,high=1.000000000000001)
        random_drop3 = np.random.uniform(low=0.985+lower_bound,high=1.000000000000001)

        for batch_size_index in range(0,len(train_batch),batch_size//2):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size//2]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size//2]

            print(current_batch.shape)
            print(current_batch_label.shape)

            # resize the image and perform augmentation and standard normalization
            current_batch = np.asarray([cv2.resize(x.astype(np.float32),(126,126),interpolation=cv2.INTER_LANCZOS4) for x in current_batch ])
            images_aug = seq.augment_images(current_batch.astype(np.float32))

            for x in range(5):
                plt.imshow(current_batch[x,:,:,:].astype(int))
                plt.show()
                plt.imshow(images_aug[x,:,:,:].astype(int))
                plt.show()
                


            current_batch = np.vstack((current_batch,images_aug)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch[:,:,:,0]  = (current_batch[:,:,:,0] - current_batch[:,:,:,0].mean(axis=0)) / ( current_batch[:,:,:,0].std(axis=0)+ 1e-20)
            current_batch[:,:,:,1]  = (current_batch[:,:,:,1] - current_batch[:,:,:,1].mean(axis=0)) / ( current_batch[:,:,:,1].std(axis=0)+ 1e-20)
            current_batch[:,:,:,2]  = (current_batch[:,:,:,2] - current_batch[:,:,:,2].mean(axis=0)) / ( current_batch[:,:,:,2].std(axis=0)+ 1e-20)
            current_batch,current_batch_label  = shuffle(current_batch,current_batch_label)
            # online data augmentation here and standard normalization
            print(current_batch.shape)
            print(current_batch_label.shape)
            sys.exit()

            sess_result = sess.run([cost,accuracy,correct_prediction,auto_train],feed_dict={x:current_batch,y:current_batch_label,
            iter_variable:iter,learning_rate_dynamic:learning_rate,batch_size_dynamic:current_batch.shape[0],
            droprate1:0.9,droprate2:1.0,droprate3:1.0})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]
            sess_result = sess.run([cost,accuracy,correct_prediction],
            feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,learning_rate_dynamic:learning_rate,
            droprate1:1.0,droprate2:1.0,droprate3:1.0,batch_size_dynamic:current_batch.shape[0]})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n----------")
            print("Learning Rate : ", learning_rate * (1.0/(1.0+learnind_rate_decay*iter)) )
            print("decay_dilated_rate : ", proportion_rate * (1.0/(1.0+decay_rate*iter)) )
            print("decouple_weight : ", decouple_weight )
            print("Drop 1 : ",random_drop1," Drop 2: ",random_drop2," Drop 3: ",random_drop3)
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size//2)),' Current Acc: ', train_acca/(len(train_batch)/(batch_size//2) ),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', test_acca/(len(test_batch)/batch_size),end='\n')
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