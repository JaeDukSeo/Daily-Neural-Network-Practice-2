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
def d_tf_LRelu(x): return tf.cast(tf.greater(x,0),tf.float32) + tf.cast(tf.less_equal(x,0),tf.float32) *x* 0.2

def tf_iden(x): return x
def d_tf_iden(x): return 1.0

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

# make class 
class PCNN():
    
    def __init__(self,ker,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.05))
        self.b = tf.Variable(tf.truncated_normal([],stddev=0.05))
        self.act,self.d_act = act,d_act

    def feedforward(self,input,mask ,stride=1,batch=True):
        self.input  = input * mask
        self.layer  =  tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding='SAME')/ tf.reduce_sum(mask) + self.b

        if batch: self.layer  = tf.nn.batch_normalization(self.layer,mean=0.0,variance=1.0,variance_epsilon=1e-8,scale=True,offset=True)

        def f1():
            return self.act(self.layer)
        def f2():
            return tf.zeros_like( self.layer )

        self.layer_return = tf.cond(tf.greater(tf.reduce_sum(mask), 0.0), true_fn=f1, false_fn=f1)
        return self.layer_return

class MCNN():
    def __init__(self,ker,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.ones([ker,ker,in_c,out_c]))
        self.b = tf.Variable(tf.zeros([]))
        self.act,self.d_act = act,d_act

    def feedforward(self,input,stride=1,batch=True):
        self.input  = input 
        self.layer  = tf.nn.conv2d(self.input,self.w,strides = [1,stride,stride,1],padding='SAME') + self.b
        # if batch: self.layer  = tf.nn.batch_normalization(self.layer,mean=0.0,variance=1.0,variance_epsilon=1e-8,scale=True,offset=True)
        def f1():
            return self.layer
        def f2():
            return tf.zeros_like( self.layer )

        self.layer_return = tf.cond(tf.greater(tf.reduce_sum(input), 0.0), true_fn=f1, false_fn=f1)
        return self.layer_return




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

train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0))
train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0))
train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0))

# generate random box as noises
num_imgs = 100
img_size = 128
min_rect_size = 10
max_rect_size = 30
num_objects = 2

imgs = np.ones((num_imgs, img_size, img_size,3))
for i_img in range(num_imgs):
    for i_object in range(num_objects):
        if np.random.choice([True, False]):
            width, height = np.random.randint(min_rect_size, max_rect_size, size=2)
            x = np.random.randint(0, img_size - width)
            y = np.random.randint(0, img_size - height)
            imgs[i_img, x:x+width, y:y+height,0] = 0
            imgs[i_img, x:x+width, y:y+height,1] = 0
            imgs[i_img, x:x+width, y:y+height,2] = 0
        else:
            size = np.random.randint(min_rect_size, max_rect_size)
            x, y = np.random.randint(0, img_size - size, size=2)
            mask = np.tril_indices(size)
            imgs[i_img, x + mask[0], y + mask[1],0] =0
            imgs[i_img, x + mask[0], y + mask[1],1] =0
            imgs[i_img, x + mask[0], y + mask[1],2] =0
            

train_mask= imgs






# class
l1x = PCNN(7,3,16,tf_Relu,d_tf_Relu)
l1m = MCNN(7,3,1,tf_Relu,d_tf_Relu)

l2x = PCNN(5,16,32,tf_Relu,d_tf_Relu)
l2m = MCNN(5,1,1,tf_Relu,d_tf_Relu)

l3x = PCNN(3,32,64,tf_Relu,d_tf_Relu)
l3m = MCNN(3,1,1,tf_Relu,d_tf_Relu)

l4x = PCNN(3,64,128,tf_Relu,d_tf_Relu)
l4m = MCNN(3,1,1,tf_Relu,d_tf_Relu)

# up
l5x = PCNN(3,192,64,tf_LRelu,d_tf_LRelu)
l5m = MCNN(3,1,1,tf_LRelu,d_tf_LRelu)

l6x = PCNN(3,96,32,tf_LRelu,d_tf_LRelu)
l6m = MCNN(3,1,1,tf_LRelu,d_tf_LRelu)

l7x = PCNN(3,48,16,tf_LRelu,d_tf_LRelu)
l7m = MCNN(3,1,1,tf_LRelu,d_tf_LRelu)

l8x = PCNN(3,19,3,tf_iden,d_tf_LRelu)
l8m = MCNN(3,1,1,tf_iden,d_tf_LRelu)






# hyper
num_epoch = 500
learing_rate = 0.00002
batch_size = 5
print_size = 10

beta_1,beta_2 = 0.9,0.999
adam_e = 1e-8




# graphs
x = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)
x_mask = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)

layer1_x = l1x.feedforward(x,x_mask,stride=2,batch=False)
layer1_m = l1m.feedforward(x_mask,stride=2,batch=False)

layer2_x = l2x.feedforward(layer1_x,layer1_m,stride=2)
layer2_m = l2m.feedforward(layer1_m,stride=2)

layer3_x = l3x.feedforward(layer2_x,layer2_m,stride=2)
layer3_m = l3m.feedforward(layer2_m,stride=2)

layer4_x = l4x.feedforward(layer3_x,layer3_m,stride=2)
layer4_m = l4m.feedforward(layer3_m,stride=2)

layer5Upsamplex = tf.image.resize_images(layer4_x,size=[layer4_x.shape[1]*2,layer4_x.shape[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
layer5Upsamplemask = tf.image.resize_images(layer4_m,size=[layer4_m.shape[1]*2,layer4_m.shape[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
layer5Input = tf.concat([layer5Upsamplex,layer3_x],axis=3)
# layer5Input_mask = tf.concat([layer5Upsamplemask,layer3_m],axis=3)
layer5_x = l5x.feedforward(layer5Input,layer5Upsamplemask,stride=1)
layer5_m = l5m.feedforward(layer5Upsamplemask,stride=1)

layer6Upsamplex = tf.image.resize_images(layer5_x,size=[layer5_x.shape[1]*2,layer5_x.shape[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
layer6Upsamplemask = tf.image.resize_images(layer5_m,size=[layer5_m.shape[1]*2,layer5_m.shape[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
layer6Input = tf.concat([layer6Upsamplex,layer2_x],axis=3)
# layer6Input_mask = tf.concat([layer6Upsamplemask,layer2_m],axis=3)
layer6_x = l6x.feedforward(layer6Input,layer6Upsamplemask,stride=1)
layer6_m = l6m.feedforward(layer6Upsamplemask,stride=1)

layer7Upsamplex = tf.image.resize_images(layer6_x,size=[layer6_x.shape[1]*2,layer6_x.shape[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
layer7Upsamplemask = tf.image.resize_images(layer6_m,size=[layer6_m.shape[1]*2,layer6_m.shape[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
layer7Input = tf.concat([layer7Upsamplex,layer1_x],axis=3)
# layer7Input_mask = tf.concat([layer7Upsamplemask,layer1_m],axis=3)
layer7_x = l7x.feedforward(layer7Input,layer7Upsamplemask,stride=1)
layer7_m = l7m.feedforward(layer7Upsamplemask,stride=1)

layer8Upsamplex = tf.image.resize_images(layer7_x,size=[layer7_x.shape[1]*2,layer7_x.shape[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
layer8Upsamplemask = tf.image.resize_images(layer7_m,size=[layer7_m.shape[1]*2,layer7_m.shape[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
layer8Input = tf.concat([layer8Upsamplex,x],axis=3)
# layer8Input_mask = tf.concat([layer8Upsamplemask,x_mask],axis=3)
layer8_x = l8x.feedforward(layer8Input,layer8Upsamplemask,stride=1)
layer8_m = l8m.feedforward(layer8Upsamplemask,stride=1)

loss_hole = tf.reduce_mean(tf.abs((1-x_mask) * (layer8_x-y) ))
loss_val = 6* tf.reduce_mean(tf.abs(x_mask * (layer8_x-y) ))

# --- auto train ---
auto_train = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(loss_hole+loss_val)

# start the session
with tf.Session() as sess : 

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        for current_batch_index in range(0,len(train_images),batch_size):
            current_batch_image = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_mask = train_mask[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([loss_hole,loss_val,auto_train],feed_dict={x:current_batch_image*current_batch_mask,y:current_batch_image,x_mask:current_batch_mask})
            print("Iter: ",iter, ' cost 1: ',sess_results[0],' cost 2: ',sess_results[1],end='\r')

        if iter % print_size == 0 :
            random_data_index = np.random.randint(len(train_images))
            current_batch_image = np.expand_dims(train_images[random_data_index,:,:,:],0)
            current_batch_mask = np.expand_dims(train_mask[random_data_index,:,:,:],0)
            sess_results = sess.run([layer8_x],feed_dict={x:current_batch_image*current_batch_mask,y:current_batch_image,x_mask:current_batch_mask})

            plt.figure()
            plt.imshow(np.squeeze(current_batch_image[0,:,:,:]))
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' Original')
            plt.savefig('training/'+str(iter)+'a_'+str(iter)+' Original Image .png')

            plt.figure()
            plt.imshow(np.squeeze(current_batch_mask[0,:,:,:]))
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' Mask ')
            plt.savefig('training/'+str(iter)+'b_'+str(iter)+' Mask Image .png')

            plt.figure()
            plt.imshow(1-np.squeeze(current_batch_mask[0,:,:,:]))
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' Mask ')
            plt.savefig('training/'+str(iter)+'b_'+str(iter)+' Mask2 Image .png')

            plt.figure()
            plt.imshow( np.squeeze(current_batch_mask[0,:,:,:] * current_batch_image[0,:,:,:])   )
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' Mask Applied image')
            plt.savefig('training/'+str(iter)+'c_'+str(iter)+' Mask Applied .png')

            plt.figure()
            plt.imshow(np.squeeze(sess_results[0][0,:,:,:]))
            plt.axis('off')
            plt.title('epoch_'+str(iter)+" Recovered Image ")
            plt.savefig('training/'+str(iter)+'d_'+str(iter)+' Recovered image.png')

            plt.close('all')
            print('\n-----------------------------------------------\n')




        if iter == num_epoch:
            
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