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

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf.log(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

# make class 
class CNNLayer():
    
    def __init__(self,ker,in_c,out_c,act,d_act):
        
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.005))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,stride=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding='SAME')
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient,stride=1):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,filter_sizes = self.w.shape,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        grad_pass  = tf.nn.conv2d_backprop_input(
            input_sizes=[batch_size] + list(self.input.shape[1:]),filter = self.w ,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        update_w = []

        update_w.append(
            tf.assign( self.m,self.m*beta_1 + (1-beta_1) * grad   )
        )
        update_w.append(
            tf.assign( self.v,self.v*beta_2 + (1-beta_2) * grad ** 2   )
        )

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,update_w

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

train_images = np.zeros(shape=(100,150,150,3))
train_labels = np.zeros(shape=(100,150,150,3))

for file_index in range(len(train_data)):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='RGB'),(150,150))
    train_labels[file_index,:,:]   = imresize(imread(train_data_gt[file_index],mode='RGB'),(150,150))

train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0))
train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0))
train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0))
train_labels[:,:,:,0]  = (train_labels[:,:,:,0] - train_labels[:,:,:,0].min(axis=0)) / (train_labels[:,:,:,0].max(axis=0) - train_labels[:,:,:,0].min(axis=0))

X = shuffle(train_images)
s_images = X[:50,:,:,:]
c_images = X[50:,:,:,:]

# hyper
num_epoch = 3000
learing_rate = 0.0002
batch_size = 5
print_size = 100

networ_beta = 1.0

beta_1,beta_2 = 0.9,0.999
adam_e = 1e-8

# init class
prep_net1 = CNNLayer(3,3,50,tf_Relu,d_tf_Relu)
prep_net2 = CNNLayer(3,50,50,tf_Relu,d_tf_Relu)
prep_net3 = CNNLayer(3,50,50,tf_Relu,d_tf_Relu)
prep_net4 = CNNLayer(3,50,50,tf_Relu,d_tf_Relu)
prep_net5 = CNNLayer(3,50,3,tf_Relu,d_tf_Relu)

hide_net1 = CNNLayer(4,6,50,tf_Relu,d_tf_Relu)
hide_net2 = CNNLayer(4,50,50,tf_Relu,d_tf_Relu)
hide_net3 = CNNLayer(4,50,50,tf_Relu,d_tf_Relu)
hide_net4 = CNNLayer(4,50,50,tf_Relu,d_tf_Relu)
hide_net5 = CNNLayer(4,50,3,tf_Relu,d_tf_Relu)

reve_net1 = CNNLayer(5,3,50,tf_Relu,d_tf_Relu)
reve_net2 = CNNLayer(5,50,50,tf_Relu,d_tf_Relu)
reve_net3 = CNNLayer(5,50,50,tf_Relu,d_tf_Relu)
reve_net4 = CNNLayer(5,50,50,tf_Relu,d_tf_Relu)
reve_net5 = CNNLayer(5,50,3,tf_Relu,d_tf_Relu)

# make graph
Secret = tf.placeholder(shape=[None,150,150,3],dtype=tf.float32)
Cover = tf.placeholder(shape=[None,150,150,3],dtype=tf.float32)

prep_layer1 = prep_net1.feedforward(Secret)
prep_layer2 = prep_net2.feedforward(prep_layer1)
prep_layer3 = prep_net3.feedforward(prep_layer2)
prep_layer4 = prep_net4.feedforward(prep_layer3)
prep_layer5 = prep_net5.feedforward(prep_layer4)

hide_Input = tf.concat([Cover,prep_layer5],axis=3)
hide_layer1 = hide_net1.feedforward(hide_Input)
hide_layer2 = hide_net2.feedforward(hide_layer1)
hide_layer3 = hide_net3.feedforward(hide_layer2)
hide_layer4 = hide_net4.feedforward(hide_layer3)
hide_layer5 = hide_net5.feedforward(hide_layer4)

reve_layer1 = reve_net1.feedforward(hide_layer5)
reve_layer2 = reve_net2.feedforward(reve_layer1)
reve_layer3 = reve_net3.feedforward(reve_layer2)
reve_layer4 = reve_net4.feedforward(reve_layer3)
reve_layer5 = reve_net5.feedforward(reve_layer4)

cost_1 = tf.reduce_mean(tf.square(hide_layer5 - Cover))
cost_2 = tf.reduce_mean(tf.square(reve_layer5 - Secret))


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