import tensorflow as tf
import numpy as np,sys,os
from numpy import float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize

np.random.seed(678)
tf.set_random_seed(678)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),tf.float32)
def tf_LRelu(x): return tf.nn.leaky_relu(x)
def d_tf_LRelu(x): return tf.cast(tf.greater(x,0),tf.float32) + tf.cast(tf.less_equal(x,0),tf.float32) *x* 0.2
def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))
def tf_log(x): return tf.nn.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1-tf_log(x))

# convolution layer
class CNNLayer():
    
    def __init__(self,ker,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.02))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
    def getw(self): return [self.w]

    def feedforward(self,input,stride=1,batch_norm=True,padding_val='SAME',mean_pooling=True):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding=padding_val)
        if batch_norm: self.layer = tf.nn.batch_normalization(self.layer,mean=0.0,variance=1.0,variance_epsilon=1e-8,scale=True,offset=True)
        if mean_pooling: self.layer = tf.nn.avg_pool(self.layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
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
# data_location = "../../Dataset/salObj/datasets/imgs/pascal/"
# train_data = []  # create an empty list
# for dirName, subdirList, fileList in sorted(os.walk(data_location)):
#     for filename in fileList:
#         if ".jpg" in filename.lower() :
#             train_data.append(os.path.join(dirName,filename))

# data_location =  "../../Dataset/salObj/datasets/masks/pascal/"
# train_data_gt = []  # create an empty list
# for dirName, subdirList, fileList in sorted(os.walk(data_location)):
#     for filename in fileList:
#         if ".png" in filename.lower() :
#             train_data_gt.append(os.path.join(dirName,filename))

# train_images = np.zeros(shape=(850,256,256,3))
# train_labels = np.zeros(shape=(850,256,256,1))

# for file_index in range(len(train_data)-1):
#     train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='RGB'),(256,256))
#     train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(256,256)),axis=3)

# train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0)+ 1e-10)
# train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0)+ 1e-10)
# train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0)+ 1e-10)
# train_labels[:,:,:,0]  = (train_labels[:,:,:,0] - train_labels[:,:,:,0].min(axis=0)) / (train_labels[:,:,:,0].max(axis=0) - train_labels[:,:,:,0].min(axis=0)+ 1e-10)

# hyper
num_epoch = 100
learing_rate = 0.001
batch_size = 10

# define class
l1_e = CNNLayer(3,3,16,tf_Relu,d_tf_Relu)
l2_e = CNNLayer(3,16,32,tf_Relu,d_tf_Relu)
l3_e = CNNLayer(3,32,64,tf_Relu,d_tf_Relu)
l4_e = CNNLayer(3,64,1,tf_Relu,d_tf_Relu)

l1_match = CNNLayer(3,1,1,tf_Relu,d_tf_Relu)
l2_match = CNNLayer(3,64,1,tf_Relu,d_tf_Relu)
l3_match = CNNLayer(3,32,1,tf_Relu,d_tf_Relu)
l4_match = CNNLayer(3,16,1,tf_Relu,d_tf_Relu)

l1_d = CNNLayer(3,2,1,tf_Relu,d_tf_Relu)
l2_d = CNNLayer(3,2,1,tf_Relu,d_tf_Relu)
l3_d = CNNLayer(3,2,1,tf_Relu,d_tf_Relu)
l4_d = CNNLayer(3,2,1,tf_Relu,d_tf_Relu)

# graph
x = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)

y1 = tf.placeholder(shape=[None,8,8,1],dtype=tf.float32)
y2 = tf.placeholder(shape=[None,16,16,1],dtype=tf.float32)
y3 = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32)
y4 = tf.placeholder(shape=[None,64,64,1],dtype=tf.float32)
y5 = tf.placeholder(shape=[None,128,128,1],dtype=tf.float32)

layer1 = l1_e.feedforward(x)
layer2 = l2_e.feedforward(layer1)
layer3 = l3_e.feedforward(layer2)
layer4 = l4_e.feedforward(layer3)
cost1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer4,labels=y1)

layer5_Match = l1_match.feedforward(layer4,mean_pooling=False)
layer5_Input = tf.concat([layer4,layer5_Match],axis=3)
layer5 = l1_d.feedforward(layer5_Input,mean_pooling=False)
layer5_Up = tf.image.resize_images(layer5,size=[16,16],method=tf.image.ResizeMethod.BILINEAR)
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer5_Up,labels=y2))

layer6_Match = l2_match.feedforward(layer3,mean_pooling=False)
layer6_Input = tf.concat([layer5_Up,layer6_Match],axis=3)
layer6 = l2_d.feedforward(layer6_Input,mean_pooling=False)
layer6_Up = tf.image.resize_images(layer6,size=[32,32],method=tf.image.ResizeMethod.BILINEAR)
cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer6_Up,labels=y3))

layer7_Match = l3_match.feedforward(layer2,mean_pooling=False)
layer7_Input = tf.concat([layer6_Up,layer7_Match],axis=3)
layer7 = l3_d.feedforward(layer7_Input,mean_pooling=False)
layer7_Up = tf.image.resize_images(layer7,size=[64,64],method=tf.image.ResizeMethod.BILINEAR)
cost4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer7_Up,labels=y4))

layer8_Match = l4_match.feedforward(layer1,mean_pooling=False)
layer8_Input = tf.concat([layer7_Up,layer8_Match],axis=3)
layer8 = l4_d.feedforward(layer8_Input,mean_pooling=False)
layer8_Up = tf.image.resize_images(layer8,size=[128,128],method=tf.image.ResizeMethod.BILINEAR)
cost5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer8_Up,labels=y5))

auto_train = tf.train.MomentumOptimizer(learning_rate=learing_rate,momentum=0.9).minimize(cost1+cost2+cost3+cost4+cost5)

# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        train_data,train_gt = shuffle(train_data,train_gt)
        for current_batch_index in range(0,len(train_data),batch_size):
            current_image_batch = train_data[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_mask_batch  = train_gt[current_batch_index:current_batch_index+batch_size,:,:,:]

            sess_results_d = sess.run([d_loss,auto_d_train],feed_dict={input_binary_image:current_image_batch,color_image:current_mask_batch})
            sess_results_g = sess.run([g_loss,auto_g_train],feed_dict={input_binary_image:current_image_batch,color_image:current_mask_batch})
            print("Current Iter: ",iter, " current batch: ",current_batch_index," D Cost: ",sess_results_d[0], " G Cost: ",sess_results_g[0],end='\r')

        if iter % print_size == 0:
            print("\n------------------------\n")
            test_example =   train_data[:2,:,:,:]
            test_example_gt = train_gt[:2,:,:,:]
            sess_results = sess.run([g_e_layer_final],feed_dict={input_binary_image:test_example,color_image:test_example_gt})

            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]
            test_example_gt = test_example_gt[0,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example),cmap='gray')
            plt.axis('off')
            plt.title('Original Mask ')
            plt.savefig('train_change/'+str(iter)+"a_Original_Mask.png")

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt))
            plt.axis('off')
            plt.title('Ground Truth Image')
            plt.savefig('train_change/'+str(iter)+"b_Original_Image.png")

            plt.figure()
            plt.axis('off')
            plt.imshow(np.squeeze(sess_results)   ,cmap='gray')
            plt.title("Generated Image")
            plt.savefig('train_change/'+str(iter)+"e_Generated_Image.png")

            plt.close('all')       

    # print halve test
    for current_batch_index in range(0,len(test_images),batch_size):
        test_example = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
        test_example_gt  = test_images_c[current_batch_index:current_batch_index+batch_size,:,:,:]
        sess_results = sess.run([g_e_layer_final],feed_dict={input_binary_image:test_example})

        sess_results = sess_results[0][0,:,:,:]
        test_example = test_example[0,:,:,:]
        test_example_gt = test_example_gt[0,:,:,:]

        plt.figure()
        plt.imshow(np.squeeze(test_example),cmap='gray')
        plt.axis('off')
        plt.title('Original Mask ')
        plt.savefig('gif/'+str(current_batch_index)+"a_Original_Mask.png")

        plt.figure()
        plt.imshow(np.squeeze(test_example_gt))
        plt.axis('off')
        plt.title('Ground Truth Image')
        plt.savefig('gif/'+str(current_batch_index)+"b_Original_Image.png")

        plt.figure()
        plt.axis('off')
        plt.imshow(np.squeeze(sess_results)   ,cmap='gray')
        plt.title("Generated Image")
        plt.savefig('gif/'+str(current_batch_index)+"e_Generated_Image.png")

        plt.close('all')       

    # print halve train
    for current_batch_index in range(0,len(train_data),batch_size):
        test_example = train_data[current_batch_index:current_batch_index+batch_size,:,:,:]
        test_example_gt  = train_gt[current_batch_index:current_batch_index+batch_size,:,:,:]
        sess_results = sess.run([g_e_layer_final],feed_dict={input_binary_image:test_example})

        sess_results = sess_results[0][0,:,:,:]
        test_example = test_example[0,:,:,:]
        test_example_gt = test_example_gt[0,:,:,:]

        plt.figure()
        plt.imshow(np.squeeze(test_example),cmap='gray')
        plt.axis('off')
        plt.title('Original Mask ')
        plt.savefig('final/'+str(current_batch_index)+"a_Original_Mask.png")

        plt.figure()
        plt.imshow(np.squeeze(test_example_gt))
        plt.axis('off')
        plt.title('Ground Truth Image')
        plt.savefig('final/'+str(current_batch_index)+"b_Original_Image.png")

        plt.figure()
        plt.axis('off')
        plt.imshow(np.squeeze(sess_results)   ,cmap='gray')
        plt.title("Generated Image")
        plt.savefig('final/'+str(current_batch_index)+"e_Generated_Image.png")

        plt.close('all')    

# -- end code --