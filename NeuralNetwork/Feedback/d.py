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
def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1 - tf.square(tf_tanh(x))
def tf_softmax(x): return tf.nn.softmax(x)

# data
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
training_images, training_labels, testing_images, testing_labels = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# class
class FNN():
    def __init__(self,input_dim,hidden_dim,act,d_act):
        self.w = tf.Variable(tf.truncated_normal([input_dim,hidden_dim], stddev=0.005))
        self.B = tf.Variable(tf.random_uniform([10,hidden_dim],minval=-0.5,maxval=0.5))
        self.act,self.d_act = act,d_act

    def feedforward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input 

        grad_added_d = tf.matmul(grad_part_1,self.B)
        grad_x_mid = tf.multiply(grad_added_d,grad_part_2)
        grad = tf.matmul(tf.transpose(grad_part_3),grad_x_mid)

        update_w = []
        update_w.append(tf.assign(self.w, self.w - learning_rate * grad))
        return None,update_w

# hyper
num_epoch = 301
batch_size = 100
print_size = 50
learning_rate = 0.0003
beta1,beta2,adam_e = 0.9,0.999,1e-8

proportion_rate = 1
decay_rate = 0.05

# define class
l1 = FNN(784,784,tf_tanh,d_tf_tanh)
l2 = FNN(784,784,tf_tanh,d_tf_tanh)
l3 = FNN(784,10,tf_tanh,d_tf_tanh)

# graph
x = tf.placeholder(shape=[None,784],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
decay_dilated_rate = proportion_rate / (1 + decay_rate * iter_variable)

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)
layer3_soft = tf_softmax(layer3)

grad3,grad3_up = l3.backprop(layer3_soft - y)
grad2,grad2_up = l2.backprop(layer3_soft - y)
grad1,grad1_up = l1.backprop(layer3_soft - y)
manual_backprop = grad1_up + grad2_up + grad3_up

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer3,labels=y))
correct_prediction = tf.equal(tf.argmax(layer3_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        training_images,training_labels = shuffle(training_images,training_labels )
        for batch_size_index in range(0,len(training_images),batch_size):
            current_batch = training_images[batch_size_index:batch_size_index+batch_size]
            current_batch_label = training_labels[batch_size_index:batch_size_index+batch_size]
            sess_result = sess.run([cost,accuracy,manual_backprop,correct_prediction,layer3_soft],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(testing_images),batch_size):
            current_batch = testing_images[test_batch_index:test_batch_index+batch_size]
            current_batch_label = testing_labels[test_batch_index:test_batch_index+batch_size]
            sess_result = sess.run([cost,accuracy,layer3,layer3_soft],feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter})
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
