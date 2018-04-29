import tensorflow as tf
import numpy as np,sys,os
from numpy import float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import hashlib
np.random.seed(678)
tf.set_random_seed(678)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),tf.float32)

def tf_LRelu(x): return tf.nn.leaky_relu(x)
# def d_tf_LRelu(x): return tf.cast(tf.greater(x,0),tf.float32)

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

# make class 
class PCNN_L():
    
    def __init__(self,ker,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.005))
        self.b = tf.Variable(tf.truncated_normal([],stddev=0.005))
        self.act,self.d_act = act,d_act

    def feedforward(self,input,mask ,stride=1):
        self.input  = input * mask
        self.layer  = tf.nn.conv2d(self.input,self.w,strides = [1,stride,stride,1],padding='SAME') + self.b
        self.layerA = self.act(self.layer)
        return self.layerA

class MCNN_L():
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.ones([ker,ker,in_c,out_c]))
        self.b = tf.Variable(tf.zeros([]))

    def feedforward(self,input,stride=1):
        self.input  = input 
        self.layer  = tf.nn.conv2d(self.input,self.w,strides = [1,stride,stride,1],padding='SAME') + self.b
        return self.layer

class PCNN_R():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.005))
        self.b = tf.Variable(tf.truncated_normal([],stddev=0.005))
        self.act,self.d_act = act,d_act

    def feedforward(self,input,stride=1,dilate=1,output=1):
        self.input  = input
        current_shape_size = input.shape

        self.layer = tf.nn.conv2d_transpose(input,self.w,output_shape=[batch_size] + [int(current_shape_size[1].value*2),int(current_shape_size[2].value*2),int(current_shape_size[3].value/2)],strides=[1,2,2,1],padding='SAME')
        self.layerA = self.act(self.layer)

        return self.layerA

class MCNN_R():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.05))

    def feedforward(self,input,stride=1,dilate=1,output=1):
        self.input  = input

        current_shape_size = input.shape

        self.layer = tf.nn.conv2d_transpose(input,self.w,
        output_shape=[batch_size] + [int(current_shape_size[1].value*2),int(current_shape_size[2].value*2),int(current_shape_size[3].value/2)],strides=[1,2,2,1],padding='SAME')
        self.layerA = tf_relu(self.layer)
        return self.layerA


# data
data_location = "../../Dataset/Semanticdataset100/image/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".jpg" in filename.lower() :
            train_data.append(os.path.join(dirName,filename))

data_location =  "../../Dataset/Semanticdataset100/ground-truth/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".png" in filename.lower() :
            train_data_gt.append(os.path.join(dirName,filename))

train_images = np.zeros(shape=(100,128,128,3))
train_labels = np.zeros(shape=(100,128,128,1))
train_labels_inverted = np.zeros(shape=(100,128,128,1))


for file_index in range(len(train_data)):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='RGB'),(128,128))
    train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F'),(128,128)),axis=3)
    train_labels_inverted[file_index,:,:]  = np.logical_not(train_labels[file_index,:,:,:])

train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0))
train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0))
train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0))
train_labels[:,:,:,0]  = (train_labels[:,:,:,0] - train_labels[:,:,:,0].min(axis=0)) / (train_labels[:,:,:,0].max(axis=0) - train_labels[:,:,:,0].min(axis=0))

train_images = shuffle(train_images)

# class
l1x = PCNN_L(7,3,64,tf_Relu,d_tf_Relu)
l1m = MCNN_L(7,1,1)

l2x = PCNN_L(3,64,128,tf_Relu,d_tf_Relu)
l2m = MCNN_L(3,1,1)

l3x = PCNN_L(3,128,256,tf_Relu,d_tf_Relu)
l3m = MCNN_L(3,1,1)

l4x = PCNN_L(3,256,256,tf_Relu,d_tf_Relu)
l4m = MCNN_L(3,1,1)

# up
l5x = PCNN_L(7,256,512,tf_LRelu,d_tf_Relu)
l5m = MCNN_L(7,1,1)

l6x = PCNN_L(3,128,512,tf_LRelu,d_tf_Relu)
l6m = MCNN_L(3,1,1)

l7x = PCNN_L(3,64,256,tf_LRelu,d_tf_Relu)
l7m = MCNN_L(3,1,1)

l8x = PCNN_L(3,3,128,tf_LRelu,d_tf_Relu)
l8m = MCNN_L(3,1,1)


# hyper
num_epoch = 3000
learing_rate = 0.0002
batch_size = 5
print_size = 100

networ_beta = 1.0

beta_1,beta_2 = 0.9,0.999
adam_e = 1e-8

# graphs
x = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)
x_mask = tf.placeholder(shape=[None,128,128,1],dtype=tf.float32)

layer1_x = l1x.feedforward(x,x_mask,stride=2)
layer1_m = l1m.feedforward(x_mask,stride=2)

layer2_x = l2x.feedforward(layer1_x,layer1_m,stride=2)
layer2_m = l2m.feedforward(layer1_m,stride=2)

layer3_x = l3x.feedforward(layer2_x,layer2_m,stride=2)
layer3_m = l3m.feedforward(layer2_m,stride=2)

layer4_x = l4x.feedforward(layer3_x,layer3_m,stride=2)
layer4_m = l4m.feedforward(layer3_m,stride=2)

layer5Upsample = tf.image.resize_images(layer4_x,size=[layer4_x.shape[1]*2,layer4_x.shape[2]*2],
method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
layer5Input = tf.concat([layer5Upsample,layer3_x],axis=3)



print(layer4_x.shape)
print(layer4_m.shape)
print(layer5Input.shape)

sys.exit()


# --- auto train ---
auto_train = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(cost_1+cost_2)

# start the session
with tf.Session() as sess : 

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        for current_batch_index in range(0,len(s_images),batch_size):
            current_batch_s = s_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_c = c_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([cost_1,cost_2,auto_train],feed_dict={Secret:current_batch_s,Cover:current_batch_c})
            print("Iter: ",iter, ' cost 1: ',sess_results[0],' cost 2: ',sess_results[1],end='\r')

        if iter % print_size == 0 :
            random_data_index = np.random.randint(len(s_images))
            current_batch_s = np.expand_dims(s_images[random_data_index,:,:,:],0)
            current_batch_c = np.expand_dims(c_images[random_data_index,:,:,:],0)
            sess_results = sess.run([prep_layer5,hide_layer5,reve_layer5],feed_dict={Secret:current_batch_s,Cover:current_batch_c})

            plt.figure()
            plt.imshow(np.squeeze(current_batch_s[0,:,:,:]))
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' Secret')
            plt.savefig('training/'+str(iter)+'a_'+str(iter)+'Secret image.png')

            plt.figure()
            plt.imshow(np.squeeze(current_batch_c[0,:,:,:]))
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' cover')
            plt.savefig('training/'+str(iter)+'b_'+str(iter)+'cover image.png')

            plt.figure()
            plt.imshow(np.squeeze(sess_results[0][0,:,:,:] ))
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' prep image')
            plt.savefig('training/'+str(iter)+'c_'+str(iter)+'prep image.png')

            plt.figure()
            plt.imshow(np.squeeze(sess_results[1][0,:,:,:] ))
            plt.axis('off')
            plt.title('epoch_'+str(iter)+" Hidden Image ")
            plt.savefig('training/'+str(iter)+'d_'+str(iter)+'Hidden image.png')

            plt.figure()
            plt.axis('off')
            plt.imshow(np.squeeze(sess_results[2][0,:,:,:] ))
            plt.title('epoch_'+str(iter)+" Reveal  Image")
            plt.savefig('training/'+str(iter)+'e_'+str(iter)+'Reveal image.png')

            plt.close('all')
            print('\n--------------------\n')

        if iter == num_epoch-1:
            
            for final in range(len(s_images)):
                current_batch_s = np.expand_dims(s_images[final,:,:,:],0)
                current_batch_c = np.expand_dims(c_images[final,:,:,:],0)
                sess_results = sess.run([prep_layer5,hide_layer5,reve_layer5],feed_dict={Secret:current_batch_s,Cover:current_batch_c})

                # create hash table 
                hash_object = hashlib.sha512(np.squeeze(current_batch_s))
                secrete_hex_digit = hash_object.hexdigest() 

                hash_object = hashlib.sha512(np.squeeze(sess_results[1][0,:,:,:]))
                prep_hex_digit = hash_object.hexdigest() 

                plt.figure()
                plt.imshow(np.squeeze(current_batch_s[0,:,:,:]))
                plt.axis('off')
                plt.title('epoch_'+str(final)+' Secret')
                plt.savefig('gif/'+str(final)+'a_'+str(secrete_hex_digit)+'Secret image.png')

                plt.figure()
                plt.imshow(np.squeeze(current_batch_c[0,:,:,:]))
                plt.axis('off')
                plt.title('epoch_'+str(final)+' cover')
                plt.savefig('gif/'+str(final)+'b_'+str(final)+'cover image.png')

                plt.figure()
                plt.imshow(np.squeeze(sess_results[0][0,:,:,:]))
                plt.axis('off')
                plt.title('epoch_'+str(final)+' prep image')
                plt.savefig('gif/'+str(final)+'c_'+str(final)+'prep image.png')

                plt.figure()
                plt.imshow(np.squeeze(sess_results[1][0,:,:,:]))
                plt.axis('off')
                plt.title('epoch_'+str(final)+" Hidden Image ")
                plt.savefig('gif/'+str(final)+'d_'+str(prep_hex_digit)+'Hidden image.png')

                plt.figure()
                plt.axis('off')
                plt.imshow(np.squeeze(sess_results[2][0,:,:,:]))
                plt.title('epoch_'+str(final)+" Reveal  Image")
                plt.savefig('gif/'+str(final)+'e_'+str(final)+'Reveal image.png')

                plt.close('all')
# -- end code --