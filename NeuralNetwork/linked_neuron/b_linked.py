import tensorflow as tf
import numpy as np
import sys, os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
from skimage.transform import resize

np.random.seed(514)
tf.set_random_seed(678)

def tf_leaky_relu(x): return tf.nn.leaky_relu(x)
def d_leaky_tf_relu(x): return tf.cast(tf.greater(x,0),dtype=tf.float32) + tf.cast(tf.less_equal(x,0),dtype=tf.float32) * 0.2
def tf_softmax(x): return tf.nn.softmax(x)

# data
mnist = input_data.read_data_sets('../Dataset/MNIST/', one_hot=True)
x_data, training_labels, y_data, testing_labels = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
x_data = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
y_data = y_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img

training_images = np.zeros((x_data.shape[0],32,32,1)).astype(np.float32)
testing_images = np.zeros((y_data.shape[0],32,32,1)).astype(np.float32)

for i in range(len(training_images)):
    training_images[i,:,:,:] = np.expand_dims(resize(np.squeeze(x_data[i,:,:,0]),(32,32)),axis=3)
for i in range(len(testing_images)):
    testing_images[i,:,:,:] = np.expand_dims(resize(np.squeeze(y_data[i,:,:,0]),(32,32)),axis=3)

# training_images = (training_images - training_images.min(axis=0))/(training_images.max(axis=0)-training_images.min(axis=0))
# testing_images = (testing_images - testing_images.min(axis=0))/(testing_images.max(axis=0)-testing_images.min(axis=0))


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

    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.005))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerAa = tf_leaky_relu(self.layer) 
        self.layerAb = tf_leaky_relu(-1.0 * self.layer)
        self.out = tf.nn.avg_pool(tf.concat([self.layerAa ,self.layerAb],3), [ 1, 2, 2, 1 ], [1, 2, 2, 1 ], 'VALID')
        return self.out

    def backprop(self,gradient):
        
        half_shape = gradient.shape[3].value//2
        gradient = tf_repeat(gradient,[1,2,2,1])
        
        grad_part_1 = gradient 
        grad_part_2a = d_leaky_tf_relu(self.layer) 
        grad_part_2b = d_leaky_tf_relu(-1.0 * self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1[:,:,:,:half_shape] * grad_part_2b + grad_part_1[:,:,:,half_shape:] * grad_part_2a

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size] + list(grad_part_3.shape[1:]),
            filter= self.w,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))
        
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)

        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,grad_update             

# hyper
num_epoch = 100
batch_size = 50
print_size = 1
learning_rate = 0.0003

beta1,beta2 = 0.9,0.999
adam_e = 1e-8

proportion_rate = 1
decay_rate = 0.05

# define class
l1 = CNN(5,1,2)
l2 = CNN(3,4,8)
l3 = CNN(3,16,4)
l4 = CNN(3,8,2)
l5 = CNN(1,4,5)

# graph
x = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
decay_dilated_rate = proportion_rate / (1 + decay_rate * iter_variable)

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)
layer4 = l4.feedforward(layer3)
layer5 = l5.feedforward(layer4)
final_soft = tf.reshape(layer5,[batch_size,-1])
final = tf_softmax(final_soft)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_soft,labels=y))
correct_prediction = tf.equal(tf.argmax(final, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- auto train ---
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        # training_images,training_labels = shuffle(training_images,training_labels )

        for batch_size_index in range(0,len(training_images),batch_size):
            current_batch = training_images[batch_size_index:batch_size_index+batch_size,:,:,:]
            current_batch_label = training_labels[batch_size_index:batch_size_index+batch_size,:]
            sess_result = sess.run([cost,accuracy,weight_update,correct_prediction,final],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(testing_images),batch_size):
            current_batch = testing_images[test_batch_index:test_batch_index+batch_size,:,:,:]
            current_batch_label = testing_labels[test_batch_index:test_batch_index+batch_size,:]
            sess_result = sess.run([cost,accuracy,final,final_soft],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n----------")
            print('Train Current cost: ', train_cota/(len(training_images)/batch_size),' Current Acc: ', train_acca/(len(training_images)/batch_size),end='\n')
            print('Test Current cost: ', test_cota/(len(testing_images)/batch_size),' Current Acc: ', test_acca/(len(testing_images)/batch_size),end='\n')
            print("----------\n")

        train_acc.append(train_acca/(len(training_images)/batch_size))
        train_cot.append(train_cota/(len(training_images)/batch_size))

        test_acc.append(test_acca/(len(testing_images)/batch_size))
        test_cot.append(test_cota/(len(testing_images)/batch_size))

        test_cota,test_acca = 0,0
        train_cota,train_acca = 0,0

    # training done
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc)
    plt.title("Train Average Accuracy Over Time")
    plt.show()

    plt.figure()
    plt.plot(range(len(train_cot)),train_cot)
    plt.title("Train Average Cost Over Time")
    plt.show()

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc)
    plt.title("Test Average Accuracy Over Time")
    plt.show()

    plt.figure()
    plt.plot(range(len(test_cot)),test_cot)
    plt.title("Test Average Cost Over Time")
    plt.show()


# -- end code --
