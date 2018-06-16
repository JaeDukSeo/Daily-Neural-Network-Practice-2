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

# activation functions
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
    iaa.Sometimes(0.5,
        iaa.Affine(
            translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
            rotate=(-10, 10),
            scale={"x": (0.5, 1.1), "y": (0.5, 1.1)},
        )
    ),
    iaa.Fliplr(1.0), # Horizonatl flips
], random_order=True) # apply augmenters in random order

# class
class CNN():
    
    def __init__(self,k,inc,out,stddev):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=stddev))
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

# data
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
# reshape data / # rotate data
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2))
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2))

# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

test_label = test_label[:50,:]
test_batch = test_batch[:50,:,:,:]

# simple normalize
train_batch = train_batch/255.0
test_batch = test_batch/255.0

# hyper parameter
num_epoch = 8
batch_size = 50
print_size = 1

learning_rate = 0.0005
learnind_rate_decay = 0.0
beta1,beta2,adam_e = 0.9,0.9,1e-8

# define class
channel_sizes = 128
l1 = CNN(3,3,channel_sizes,stddev=0.04)
l2 = CNN(3,channel_sizes,channel_sizes,stddev=0.05)
l3 = CNN(3,channel_sizes,channel_sizes,stddev=0.06)

l4 = CNN(3,channel_sizes,channel_sizes,stddev=0.04)
l5 = CNN(3,channel_sizes,channel_sizes,stddev=0.05)
l6 = CNN(3,channel_sizes,channel_sizes,stddev=0.06)

l7 = CNN(3,channel_sizes,channel_sizes,stddev=0.06)
l8 = CNN(1,channel_sizes,channel_sizes,stddev=0.05)
l9 = CNN(1,channel_sizes,10,stddev=0.04)

all_weights = [l1.getw(),l2.getw(),l3.getw(),l4.getw(),l5.getw(),l6.getw(),l7.getw(),l8.getw(),l9.getw()]

# graph
x = tf.placeholder(shape=[batch_size,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[batch_size,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_dynamic  = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate_dynamic * (1.0/(1.0+learnind_rate_decay*iter_variable))
phase = tf.placeholder(tf.bool)

layer1 = l1.feedforward(x,padding='SAME')
layer2 = l2.feedforward(layer1,padding='SAME')
layer3 = l3.feedforward(layer2,padding='SAME')

layer4_Input = tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input,padding='SAME')
layer5 = l5.feedforward(layer4,padding='SAME')
layer6 = l6.feedforward(layer5,padding='SAME')

layer7_Input = tf.nn.avg_pool(layer6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer7 = l7.feedforward(layer7_Input,padding='SAME')
layer8 = l8.feedforward(layer7,padding='VALID')
layer9 = l9.feedforward(layer8,padding='VALID')

final_global = tf.reduce_mean(layer9,[1,2])
final_soft = tf_softmax(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y) )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

grad_prepare = tf_repeat(tf.reshape(final_soft-y,[batch_size,1,1,10]),[1,8,8,1])
grad9,grad9_up = l9.backprop(grad_prepare,learning_rate_change=learning_rate_change,padding='VALID')
grad8,grad8_up = l8.backprop(grad9,learning_rate_change=learning_rate_change,padding='VALID')
grad7,grad7_up = l7.backprop(grad8,learning_rate_change=learning_rate_change)

grad6_Input = tf_repeat(grad7,[1,2,2,1])
grad6,grad6_up = l6.backprop(grad6_Input,learning_rate_change=learning_rate_change)
grad5,grad5_up = l5.backprop(grad6,learning_rate_change=learning_rate_change)
grad4,grad4_up = l4.backprop(grad5,learning_rate_change=learning_rate_change)

grad3_Input = tf_repeat(grad4,[1,2,2,1])
grad3,grad3_up = l3.backprop(grad3_Input,learning_rate_change=learning_rate_change)
grad2,grad2_up = l2.backprop(grad3,learning_rate_change=learning_rate_change)
grad1,grad1_up = l1.backprop(grad2,learning_rate_change=learning_rate_change)

grad_update =  grad9_up + grad8_up + grad7_up + \
               grad6_up + grad5_up + grad4_up + \
               grad3_up + grad2_up + grad1_up 

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    def show_hist_of_weigt(all_weight_list,status='before'):
        fig = plt.figure()
        weight_index = 0
        for i in range(1,4):
            ax = fig.add_subplot(1,3,i)
            ax.grid(False)
            temp_weight_list = all_weight_list[weight_index:weight_index+3]
            for temp_index in range(len(temp_weight_list)):
                current_flat = temp_weight_list[temp_index].flatten()
                ax.hist(current_flat,histtype='step',bins='auto',label=str(temp_index+weight_index))
                ax.legend()
            ax.set_title('From Layer : '+str(weight_index+1)+' to '+str(weight_index+3))
            weight_index = weight_index + 3
        plt.savefig('viz/z_weights_'+str(status)+"_training.png")
        plt.close('all')

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

        for batch_size_index in range(0,len(train_batch),batch_size//2):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size//2]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size//2]

            # online data augmentation here 
            images_aug1 = seq.augment_images(current_batch.astype(np.float32))
            current_batch = np.vstack((current_batch,images_aug1)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch,current_batch_label  = shuffle(current_batch,current_batch_label)
            # online data augmentation here 

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
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
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

    # plot the training and testing graph
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig("viz/z_Case Train.png")
    plt.close('all')

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("viz/z_Case Test.png")
    plt.close('all')

    # ------- histogram of weights after training ------
    show_hist_of_weigt(sess.run(all_weights),status='After')
    # ------- histogram of weights after training ------

    # get random 50 images from the test set and vis the gradient
    test_batch = test_batch[:batch_size,:,:,:]
    test_label = test_label[:batch_size,:]

    # Def: Simple function to show 9 image with different channels
    def show_9_images(image,layer_num=None,image_num=None,channel_increase=3,alpha=None,image_index=None,gt=None,predict=None):
        image = (image-image.min())/(image.max()-image.min())
        fig = plt.figure()
        color_channel = 0
        for i in range(1,10):
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
            plt.savefig('viz/y_'+str(image_index)+ "_" +str(alpha) + "_alpha_image.png")
        else:
            plt.savefig('viz/'+str(layer_num) + "_layer_"+str(image_num)+"_image.png")
        plt.close('all')

    # ------ layer wise activation -------
    layer3_values = sess.run(layer3,feed_dict={x:test_batch})
    for immage_index in range(10):
        show_9_images(layer3_values[immage_index,:,:,:],3,immage_index)

    layer6_values = sess.run(layer6,feed_dict={x:test_batch})
    for immage_index in range(10):
        show_9_images(layer6_values[immage_index,:,:,:],6,immage_index)

    layer9_values = sess.run(layer9,feed_dict={x:test_batch})
    for immage_index in range(10):
        show_9_images(layer9_values[immage_index,:,:,:],9,immage_index,channel_increase=1)
    # ------ layer wise activation -------

    # -------- Interior Gradients -----------
    # portion of code from: https://github.com/ankurtaly/Integrated-Gradients/blob/master/attributions.ipynb
    final_prediction_argmax = None
    final_gt_argmax = None
    def gray_scale(img):
        img = np.average(img, axis=2)
        return np.transpose([img, img, img], axes=[1,2,0])

    def normalize(attrs, ptile=99):
        h = np.percentile(attrs, ptile)
        l = np.percentile(attrs, 100-ptile)
        return np.clip(attrs/max(abs(h), abs(l)), -1.0, 1.0) 

    for alpha_values in [0.01, 0.02, 0.03, 0.04,0.1, 0.5, 0.6, 0.7, 0.8, 1.0]:

        # create the counterfactual input and feed it to get the gradient
        test_batch_a = test_batch * alpha_values
        sess_result = sess.run([cost,accuracy,correct_prediction,final_soft,grad1],feed_dict={x:test_batch_a,y:test_label,iter_variable:1.0,phase:False})
        
        # get the final prediction and the ground truth
        final_prediction_argmax = list(np.argmax(sess_result[3],axis=1))[:9]
        final_gt_argmax         = list(np.argmax(test_label,axis=1))[:9]

        # get the gradients
        returned_gradient_batch = sess_result[4]
        aggregated_gradient = np.expand_dims(np.average(returned_gradient_batch,axis=3),axis=3)
        attrs = abs(np.repeat(aggregated_gradient,3,axis=3))
        attrs = np.clip(attrs/np.percentile(attrs, 99), 0,1)

        # interior grad
        interrior_grad = test_batch * attrs
        stacked_grad = interrior_grad[0,:,:,:]
        for indexing in range(1,9):
            stacked_grad = np.vstack((stacked_grad.T,interrior_grad[indexing,:,:,:].T)).T
        
        # show
        show_9_images(stacked_grad,alpha=alpha_values,gt=final_gt_argmax,predict=final_prediction_argmax,image_index='1')

        # overlay interior gradient
        image_gray = np.expand_dims(np.average(test_batch,axis=3),axis=3)[:9,:,:,:]
        grad_norm = np.expand_dims(normalize(gray_scale(aggregated_gradient[0,:,:,:])),0)
        for indexing in range(1,9):
            current_image_norm = np.expand_dims(normalize(gray_scale(aggregated_gradient[indexing,:,:,:])),0)
            grad_norm = np.vstack((grad_norm,current_image_norm))

        pos_attrs = grad_norm * (grad_norm >= 0.0)
        neg_attrs = -1.0 * grad_norm * (grad_norm < 0.0) 

        # overlayer
        red_channel = np.zeros_like(grad_norm)
        red_channel[:,:,:,0] = 1.0

        blue_channel = np.zeros_like(grad_norm)
        blue_channel[:,:,:,2] = 1.0

        attrs_mask = pos_attrs*blue_channel + neg_attrs*red_channel
        vis = 0.6*image_gray + 0.4*attrs_mask

        stacked_grad2 = vis[0,:,:,:]
        for indexing in range(1,9):
            stacked_grad2 = np.vstack((stacked_grad2.T,vis[indexing,:,:,:].T)).T

        # show
        show_9_images(stacked_grad2,alpha=alpha_values,gt=final_gt_argmax,predict=final_prediction_argmax,image_index='2')
    # -------- Interior Gradients -----------

    # -------- Intergral Gradients ----------
    base_line  = test_batch * 0.0
    difference = test_batch - base_line
    step_size = 3000

    running_example = test_batch * 0.0
    for rim in range(1,step_size+1):
        current_alpha = rim / step_size
        test_batch_a = current_alpha * test_batch
        sess_result = sess.run([cost,accuracy,correct_prediction,final_soft,grad1],feed_dict={x:test_batch_a,y:test_label,iter_variable:1.0,phase:False})
        running_example = running_example + sess_result[4]
        final_prediction_argmax = list(np.argmax(sess_result[3],axis=1))[:9]

    running_example = running_example * difference
    attrs = np.expand_dims(np.average(running_example,axis=3),axis=3)
    attrs = abs(np.repeat(attrs,3,axis=3))
    attrs = np.clip(attrs/np.percentile(attrs, 99), 0,1)

    # Intergral grad
    Intergral_grad = test_batch * attrs
    stacked_grad = Intergral_grad[0,:,:,:]
    for indexing in range(1,9):
        stacked_grad = np.vstack((stacked_grad.T,Intergral_grad[indexing,:,:,:].T)).T

    # show
    show_9_images(stacked_grad,alpha=-99,gt=final_gt_argmax,predict=final_prediction_argmax,image_index='1') 

    # overlay Intergral gradient
    image_gray = np.expand_dims(np.average(test_batch,axis=3),axis=3)[:9,:,:,:]
    grad_norm = np.expand_dims(normalize(gray_scale(running_example[0,:,:,:])),0)
    for indexing in range(1,9):
        current_image_norm = np.expand_dims(normalize(gray_scale(aggregated_gradient[indexing,:,:,:])),0)
        grad_norm = np.vstack((grad_norm,current_image_norm))

    pos_attrs = grad_norm * (grad_norm >= 0.0)
    neg_attrs = -1.0 * grad_norm * (grad_norm < 0.0) 

    # overlayer
    red_channel = np.zeros_like(grad_norm)
    red_channel[:,:,:,0] = 1.0

    blue_channel = np.zeros_like(grad_norm)
    blue_channel[:,:,:,2] = 1.0

    attrs_mask = pos_attrs*blue_channel + neg_attrs*red_channel
    vis = 0.6*image_gray + 0.4*attrs_mask

    stacked_grad2 = vis[0,:,:,:]
    for indexing in range(1,9):
        stacked_grad2 = np.vstack((stacked_grad2.T,vis[indexing,:,:,:].T)).T

    # show
    show_9_images(stacked_grad2,alpha=-99,gt=final_gt_argmax,predict=final_prediction_argmax,image_index='2')
    # -------- Intergral Gradients ----------


# -- end code --