import tensorflow as tf
import numpy as np
import sys, os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(6789)
tf.set_random_seed(678)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(x): return tf.cast(tf.greater(x,0.0),tf.float32)


# data
mnist = input_data.read_data_sets('../Dataset/MNIST/', one_hot=True)

x_data, training_labels, y_data, testing_images = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
training_images = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
testing_images = y_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img


# class
class CNN():

    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out]))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
    
    def feedforward(self,input):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerAa = tf_relu(self.layer) 
        self.layerAb = tf_relu(-1.0 * self.layer)
        self.out = tf.nn.avg_pool(tf.concat([self.layerAa ,self.layerAb],-1), [ 1, 2, 2, 1 ], [1, 2, 2, 1 ], 'VALID')
        return self.out

    def backprop(self,gradient):
        grad_part_1 = gradient 
        half_shape = gradient.shape[3]//2
        grad_part_2a = d_tf_relu(gradient[:,:,:,:half_shape]) 
        grad_part_2b = d_tf_relu(gradient[:,:,:,half_shape:])
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1[:,:,:,:half_shape] * grad_part_2a
                                + grad_part_1[:,:,:,half_shape:] * grad_part_2b)

        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = grad_part_3.shape,
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
batch_size = 200
print_size = 2
learning_rate = 0.001

beta_1,beta_2 = 0.9,0.9999
adam_e = 1e-9

# define class
l1 = CNN(5,3,4)
l2 = CNN(3,8,16)
l3 = CNN(3,32,8)
l4 = CNN(1,16,4)
l5 = CNN(1,8,5)

# graph
x = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)
layer4 = l4.feedforward(layer3)
layer5 = l5.feedforward(layer4)

final_soft = tf.reshape(layer5,[batch_size,-1])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_soft,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# -- auto train --
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# -- back prop -- 

# session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    total_cost_track = 0
    cost_over_time = []
    avg_over_time = []
    avg_accuracy = 0
    avg_cost = 0

    for iter in range(num_epoch):

        for batch_size_index in range(0,len(training_images),batch_size):
            current_batch = training_images[batch_size_index:batch_size_index+batch_size,:,:,:]
            current_batch_label = training_labels[batch_size_index:batch_size_index+batch_size,:]
            sess_result = sess.run([cost,accuracy,auto_train],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            total_cost_track = total_cost_track + sess_result[0]

        for test_batch_index in range(0,len(testing_images),batch_size):
            current_batch = testing_images[test_batch_index:test_batch_index+batch_size,:,:,:]
            current_batch_label = testing_labels[test_batch_index:test_batch_index+batch_size,:]
            sess_result = sess.run([cost,accuracy,final_soft],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            avg_accuracy = sess_result[1] + avg_accuracy
            avg_cost     = sess_result[0] + avg_cost

        if iter % print_size==0:
            print("\n----------")
            print("Current Total Cost: ", total_cost_track/(len(training_images)/batch_size))
            print('Test Current cost: ', avg_cost/(len(testing_images)/batch_size),' Current Acc: ', avg_accuracy/(len(testing_images)/batch_size),end='\n')
            print("----------\n")
            avg_accuracy = 0
            avg_cost = 0

        avg_over_time.append(avg_accuracy/(len(testing_images)/batch_size))
        cost_over_time.append(avg_cost/(len(testing_images)/batch_size))
        total_cost_track = 0
        avg_accuracy = 0
        avg_cost = 0

    # training done
    plt.figure()
    plt.plot(range(len(avg_over_time)),avg_over_time)
    plt.title("Average Accuracy Over Time")
    plt.show()

    plt.figure()
    plt.plot(range(len(cost_over_time)),cost_over_time)
    plt.title("Average Cost Over Time")
    plt.show()


# -- end code --
