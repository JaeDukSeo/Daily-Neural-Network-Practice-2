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
x_data, training_labels, y_data, testing_labels = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
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
batch_size = 100
print_size = 1
learning_rate = 0.002

beta_1,beta_2 = 0.9,0.9999
adam_e = 1e-9

# define class
l1 = CNN(5,1,4)
l2 = CNN(3,8,16)
l3 = CNN(3,32,8)
l4 = CNN(1,16,5)

# graph
x = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2)
layer4 = l4.feedforward(layer3)

final_soft = tf.reshape(layer4,[batch_size,-1])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_soft,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# -- back prop -- 
grad4,grad4u = l4.backprop(tf.reshape(final_soft-y,[batch_size,1,1,10] ) )
grad3,grad3u = l3.backprop(grad4)
grad2,grad2u = l2.backprop(grad3)
grad1,grad1u = l1.backprop(grad2)
weight_update = grad1u + grad2u + grad3u + grad4u

# session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        for batch_size_index in range(0,len(training_images),batch_size):
            current_batch = training_images[batch_size_index:batch_size_index+batch_size,:,:,:]
            current_batch_label = training_labels[batch_size_index:batch_size_index+batch_size,:]
            sess_result = sess.run([cost,accuracy,weight_update],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            

        for test_batch_index in range(0,len(testing_images),batch_size):
            current_batch = testing_images[test_batch_index:test_batch_index+batch_size,:,:,:]
            current_batch_label = testing_labels[test_batch_index:test_batch_index+batch_size,:]
            sess_result = sess.run([cost,accuracy,final_soft],feed_dict={x:current_batch,y:current_batch_label})
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
    plt.plot(range(len(avg_over_time)),avg_over_time)
    plt.title("Average Accuracy Over Time")
    plt.show()

    plt.figure()
    plt.plot(range(len(cost_over_time)),cost_over_time)
    plt.title("Average Cost Over Time")
    plt.show()


# -- end code --
