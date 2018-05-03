import tensorflow as tf
import numpy as np,sys,os
from numpy import float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize

np.random.seed(6783)
tf.set_random_seed(6785)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),tf.float32)
def tf_LRelu(x): return tf.nn.leaky_relu(x)
def d_tf_LRelu(x): return tf.cast(tf.greater(x,0),tf.float32) + tf.cast(tf.less_equal(x,0),tf.float32) *x* 0.2
def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))
def tf_log(x): return tf.nn.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1-tf_log(x))
def tf_softmax(x): return tf.nn.softmax(x)

# convolution layer
class CNNLayer():
    
    def __init__(self,ker,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.5))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return [self.w]

    def feedforward(self,input,stride=1,padding_val='SAME',batch_norm=True,mean_pooling=True,no_activation=False):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding=padding_val)
        if batch_norm: self.layer = tf.nn.batch_normalization(self.layer,mean=0.0,variance=1.0,variance_epsilon=1e-8,scale=True,offset=True)
        if mean_pooling: self.layer = tf.nn.avg_pool(self.layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
        if no_activation: return self.layer
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
data_location = "../../Dataset/salObj/datasets/imgs/pascal/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".jpg" in filename.lower() :
            train_data.append(os.path.join(dirName,filename))

data_location =  "../../Dataset/salObj/datasets/masks/pascal/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".png" in filename.lower() :
            train_data_gt.append(os.path.join(dirName,filename))

train_images = np.zeros(shape=(850,128,128,3))
train_labels = np.zeros(shape=(850,128,128,1))

for file_index in range(len(train_data)-1):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='RGB'),(128,128))
    train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(128,128)),axis=3)
train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0)+1e-10)
train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0)+1e-10)
train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0)+1e-10)
train_labels[:,:,:,0]  = (train_labels[:,:,:,0] - train_labels[:,:,:,0].min(axis=0)) / (train_labels[:,:,:,0].max(axis=0) - train_labels[:,:,:,0].min(axis=0)+1e-10)




# hyper
num_epoch = 500
learing_rate = 0.00001
batch_size = 10
print_size = 5

# define 
l1_e = CNNLayer(3,3,16,tf_Relu,d_tf_Relu)
l2_e = CNNLayer(3,16,32,tf_Relu,d_tf_Relu)
l3_e = CNNLayer(3,32,64,tf_Relu,d_tf_Relu)
l4_e = CNNLayer(3,64,128,tf_Relu,d_tf_Relu)

l5_match = CNNLayer(3,128,1,tf_Relu,d_tf_Relu)
l6_match = CNNLayer(3,64,1,tf_Relu,d_tf_Relu)
l7_match = CNNLayer(3,32,1,tf_Relu,d_tf_Relu)
l8_match = CNNLayer(3,16,1,tf_Relu,d_tf_Relu)

l5_d = CNNLayer(3,128,1,tf_Relu,d_tf_Relu)
l6_d = CNNLayer(3,65,1,tf_Relu,d_tf_Relu)
l7_d = CNNLayer(3,33,1,tf_Relu,d_tf_Relu)
l8_d = CNNLayer(3,17,1,tf_Relu,d_tf_Relu)
l9_d = CNNLayer(3,4,1,tf_Relu,d_tf_Relu)

# graph
x = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)

y5 = tf.placeholder(shape=[None,128,128,1],dtype=tf.float32)
y4 = tf.image.resize_images(y5,size=[64,64])
y3 = tf.image.resize_images(y5,size=[32,32])
y2 = tf.image.resize_images(y5,size=[16,16])
y1 = tf.image.resize_images(y5,size=[8,8])

# encode
layer1 = l1_e.feedforward(x)
layer2 = l2_e.feedforward(layer1)
layer3 = l3_e.feedforward(layer2)
layer4 = l4_e.feedforward(layer3)

layer5 = l5_d.feedforward(layer4,mean_pooling=False)
layer5_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer5,labels=y1))

layer6_Upsample = tf.image.resize_images(layer5,size=[16,16],method=tf.image.ResizeMethod.BILINEAR)
layer6_Input = tf.concat([layer6_Upsample,layer3],axis=3)
layer6 = l6_d.feedforward(layer6_Input,mean_pooling=False)
layer6_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer6,labels=y2))

layer7_Upsample = tf.image.resize_images(layer6,size=[32,32],method=tf.image.ResizeMethod.BILINEAR)
layer7_Input = tf.concat([layer7_Upsample,layer2],axis=3)
layer7 = l7_d.feedforward(layer7_Input,mean_pooling=False)
layer7_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer7,labels=y3))

layer8_Upsample = tf.image.resize_images(layer7,size=[64,64],method=tf.image.ResizeMethod.BILINEAR)
layer8_Input = tf.concat([layer8_Upsample,layer1],axis=3)
layer8 = l8_d.feedforward(layer8_Input,mean_pooling=False)
layer8_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer8,labels=y4))

layer9_Upsample = tf.image.resize_images(layer8,size=[128,128],method=tf.image.ResizeMethod.BILINEAR)
layer9_Input = tf.concat([layer9_Upsample,x],axis=3)
layer9 = l9_d.feedforward(layer9_Input,mean_pooling=False)
layer9_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer9,labels=y5))





layer4Match = l0_match.feedforward(layer4,mean_pooling=False)
cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer4Match,labels=y1))

# layer5_Match = l1_match.feedforward(layer3,mean_pooling=False)
layer5_Input = tf.concat([layer4,layer4Match],axis=3)
layer5 = l1_d.feedforward(layer5_Input,mean_pooling=False)
layer5_Up = tf.image.resize_images(layer5,size=[16,16],method=tf.image.ResizeMethod.BILINEAR)
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer5_Up,labels=y2))

layer6_Match = l2_match.feedforward(layer2,mean_pooling=False)
layer6_Input = tf.concat([layer5_Up,layer6_Match],axis=3)
layer6 = l2_d.feedforward(layer6_Input,mean_pooling=False)
layer6_Up = tf.image.resize_images(layer6,size=[32,32],method=tf.image.ResizeMethod.BILINEAR)
cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer6_Up,labels=y3))

layer7_Match = l3_match.feedforward(layer1,mean_pooling=False)
layer7_Input = tf.concat([layer6_Up,layer7_Match],axis=3)
layer7 = l3_d.feedforward(layer7_Input,mean_pooling=False)
layer7_Up = tf.image.resize_images(layer7,size=[64,64],method=tf.image.ResizeMethod.BILINEAR)
cost4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer7_Up,labels=y4))

layer8_Match = l4_match.feedforward(x,mean_pooling=False)
layer8_Input = tf.concat([layer7_Up,layer8_Match],axis=3)
layer8 = l4_d.feedforward(layer8_Input,mean_pooling=False)
layer8_Up = tf.image.resize_images(layer8,size=[128,128],method=tf.image.ResizeMethod.BILINEAR)
cost5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer8_Up,labels=y5))

stage5_image,stage4_image,stage3_image,stage2_image,stage1_image = tf_softmax(layer8_Up),tf_softmax(layer7_Up),tf_softmax(layer6_Up),tf_softmax(layer5_Up),tf_softmax(layer4)
auto_train = tf.train.MomentumOptimizer(learning_rate=learing_rate,momentum=0.9).minimize(cost1+cost2+cost3+cost4+cost5)
# auto_train = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(cost1+cost2+cost3+cost4+cost5)






# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        train_images,train_labels = shuffle(train_images,train_labels)
        for current_batch_index in range(0,len(train_data),batch_size):
            
            current_image_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_mask_batch  = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]

            sess_results = sess.run([auto_train,cost5,cost4,cost3,cost2,cost1],feed_dict={x:current_image_batch,y5:current_mask_batch})
            print("Current Iter: ",iter, " current batch: ",current_batch_index,
            ' Cost 5: ',sess_results[1],' Cost 4: ',sess_results[2],' Cost 3:',sess_results[3],' Cost 2:',sess_results[4],' Cost 1:',sess_results[5]
            ,end='\r')

        if iter % print_size == 0:
            print("\n------------------------\n")
            test_example    = train_images[:2,:,:,:]
            test_example_gt = train_labels[:2,:,:,:]
            sess_results = sess.run([stage1_image],feed_dict={x:test_example,y5:test_example_gt})

            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]
            test_example_gt = test_example_gt[0,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example))
            plt.axis('off')
            plt.title('Original Image')
            plt.savefig('train_change/'+str(iter)+"a Original Image.png")

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt),cmap='gray')
            plt.axis('off')
            plt.title('Ground Truth Mask')
            plt.savefig('train_change/'+str(iter)+"b Original Mask.png")

            plt.figure()
            plt.axis('off')
            plt.imshow(np.squeeze(sess_results),cmap='gray')
            plt.title("Generated Mask")
            plt.savefig('train_change/'+str(iter)+"d Generated Mask.png")

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt * test_example))
            plt.axis('off')
            plt.title('Ground Truth Overlayed')
            plt.savefig('train_change/'+str(iter)+"e Overlayed Mask GT.png")

            # plt.figure()
            # plt.axis('off')
            # plt.imshow(np.squeeze(sess_results*test_example),cmap='gray')
            # plt.title("Generated Overlayed")
            # plt.savefig('train_change/'+str(iter)+"f Overlayed Mask.png")

            plt.close('all')       

    # Print halve test
    # for current_batch_index in range(0,len(test_images),batch_size):
    #     test_example = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     test_example_gt  = test_images_c[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     sess_results = sess.run([g_e_layer_final],feed_dict={input_binary_image:test_example})

    #     sess_results = sess_results[0][0,:,:,:]
    #     test_example = test_example[0,:,:,:]
    #     test_example_gt = test_example_gt[0,:,:,:]

    #     plt.figure()
    #     plt.imshow(np.squeeze(test_example),cmap='gray')
    #     plt.axis('off')
    #     plt.title('Original Mask ')
    #     plt.savefig('gif/'+str(current_batch_index)+"a_Original_Mask.png")

    #     plt.figure()
    #     plt.imshow(np.squeeze(test_example_gt))
    #     plt.axis('off')
    #     plt.title('Ground Truth Image')
    #     plt.savefig('gif/'+str(current_batch_index)+"b_Original_Image.png")

    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(np.squeeze(sess_results)   ,cmap='gray')
    #     plt.title("Generated Image")
    #     plt.savefig('gif/'+str(current_batch_index)+"e_Generated_Image.png")

    #     plt.close('all')       

    # Print halve train
    # for current_batch_index in range(0,len(train_data),batch_size):
    #     test_example = train_data[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     test_example_gt  = train_gt[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     sess_results = sess.run([g_e_layer_final],feed_dict={input_binary_image:test_example})

    #     sess_results = sess_results[0][0,:,:,:]
    #     test_example = test_example[0,:,:,:]
    #     test_example_gt = test_example_gt[0,:,:,:]

    #     plt.figure()
    #     plt.imshow(np.squeeze(test_example),cmap='gray')
    #     plt.axis('off')
    #     plt.title('Original Mask ')
    #     plt.savefig('final/'+str(current_batch_index)+"a_Original_Mask.png")

    #     plt.figure()
    #     plt.imshow(np.squeeze(test_example_gt))
    #     plt.axis('off')
    #     plt.title('Ground Truth Image')
    #     plt.savefig('final/'+str(current_batch_index)+"b_Original_Image.png")

    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(np.squeeze(sess_results)   ,cmap='gray')
    #     plt.title("Generated Image")
    #     plt.savefig('final/'+str(current_batch_index)+"e_Generated_Image.png")

    #     plt.close('all')    

# -- end code --