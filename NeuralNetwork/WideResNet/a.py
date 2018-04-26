import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder


def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(x): return tf.cast(tf.greater(x,0),tf.float32)
def tf_soft(x): return tf.nn.softmax(x)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# data
mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
x_data, train_label, y_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
x_data = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
y_data = y_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img

train_batch = np.zeros((x_data.shape[0],32,32,1)).astype(np.float32)
test_batch = np.zeros((y_data.shape[0],32,32,1)).astype(np.float32)

for i in range(len(train_batch)):
    train_batch[i,:,:,:] = np.expand_dims(resize(np.squeeze(x_data[i,:,:,0]),(32,32)),axis=3)
for i in range(len(test_batch)):
    test_batch[i,:,:,:] = np.expand_dims(resize(np.squeeze(y_data[i,:,:,0]),(32,32)),axis=3)

# class
class cnn():
    
    def __init__(self,k,inc,out):
        self.w1 = tf.Variable(tf.random_normal([k,k,inc,out]))
        self.w2 = tf.Variable(tf.random_normal([k,k,out,out]))
        
        self.m,self.v = tf.Variable(tf.zeros_like(self.w1)),tf.Variable(tf.zeros_like(self.w1))

    def feedforward(self,input,resadd=True):
        self.input  = input

        self.layer1  = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        self.layerA1  = tf_relu(self.layer1) 
        self.layer2  = tf.nn.conv2d(self.layerA1,self.w2,strides=[1,1,1,1],padding='SAME')
        self.layerA2  = tf_relu(self.layer2) 

        if resadd: return self.layerA2 + input
        return self.layerA2 

    def backprop(self,gradient):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_relu(self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2

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
learning_rate = 0.00001
beta1,beta2,adame = 0.9,0.999,1e-8







# class
l1 = cnn(3,1,2)

l2_1 = cnn(3,2,2)
l2_2 = cnn(3,2,2)
l2_3 = cnn(3,2,2)
l2_4 = cnn(3,2,4)

l3_1 = cnn(3,4,4)
l3_2 = cnn(3,4,4)
l3_3 = cnn(3,4,4)
l3_4 = cnn(3,4,8)

l4_1 = cnn(3,8,8)
l4_2 = cnn(3,8,8)
l4_3 = cnn(3,8,8)
l4_4 = cnn(3,8,10)








# graph
x = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

layer1 = l1.feedforward(x,resadd=False)

layer2Input = tf.nn.max_pool(layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2_1 = l2_1.feedforward(layer2Input)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_3 = l2_3.feedforward(layer2_2)
layer2_4 = l2_4.feedforward(layer2_3,resadd=False)

layer3Input = tf.nn.avg_pool(layer2_4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3_1 = l3_1.feedforward(layer3Input)
layer3_2 = l3_2.feedforward(layer3_1)
layer3Input2 = tf.nn.max_pool(layer3_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3_3 = l3_3.feedforward(layer3Input2)
layer3_4 = l3_4.feedforward(layer3_3,resadd=False)

layer4Input = tf.nn.avg_pool(layer3_4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4_1 = l4_1.feedforward(layer4Input)
layer4_2 = l4_2.feedforward(layer4_1)
layer4Input2 = tf.nn.max_pool(layer4_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4_3 = l4_3.feedforward(layer4Input2)
layer4_4 = l4_4.feedforward(layer4_3,resadd=False)

final = tf.reshape(layer4_4,[batch_size,-1])
final_soft = tf_soft(final)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# -- auto train ---
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)










# session
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())

    train_total_cost,train_total_acc =0,0
    train_cost_overtime,train_acc_overtime = [],[]

    test_total_cost,test_total_acc = 0,0
    test_cost_overtime,test_acc_overtime = [],[]

    # start the train
    for iter in range(num_epoch):
        
        train_batch,train_label = shuffle(train_batch,train_label)

        # Train Batch
        for current_batch_index in range(0,len(train_batch),batch_size):
            current_batch = train_batch[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_label[current_batch_index:current_batch_index+batch_size,:]
            sess_results = sess.run([cost,accuracy,correct_prediction,final_soft,final,auto_train], feed_dict= {x:current_batch,y:current_batch_label})
            print("current iter:", iter, ' current batch: ', current_batch_index, " current cost: %.5f"%sess_results[0],' current acc: %.5f'%sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Batch
        for current_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = test_label[current_batch_index:current_batch_index+batch_size,:]
            sess_results = sess.run([cost,accuracy,correct_prediction], feed_dict= {x:current_batch,y:current_batch_label})
            print("current iter:", iter, ' current batch: ', current_batch_index, " current cost: %.5f"%sess_results[0],' current acc: %.5f'%sess_results[1], end='\r')
            test_total_cost = test_total_cost + sess_results[0]
            test_total_acc = test_total_acc + sess_results[1]

        # store
        train_cost_overtime.append(train_total_cost/(len(train_batch)/batch_size ) )
        train_acc_overtime.append(train_total_acc/(len(train_batch)/batch_size ) )
        test_cost_overtime.append(test_total_cost/(len(test_batch)/batch_size ) )
        test_acc_overtime.append(test_total_acc/(len(test_batch)/batch_size ) )
        
        # print
        if iter%print_size == 0:
            print('\n------ Current Iter : ',iter)
            print("Avg Train Cost: ", train_cost_overtime[-1])
            print("Avg Train Acc: ", train_acc_overtime[-1])
            print("Avg Test Cost: ", test_cost_overtime[-1])
            print("Avg Test Acc: ", test_acc_overtime[-1])
            print('-----------')      
        train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0            

# plot and save
plt.figure()
plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='y',label='Original Model')
plt.legend()
plt.savefig('og Train Cost over time')

plt.figure()
plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='y',label='Original Model')
plt.legend()
plt.savefig('og Train Acc over time')

plt.figure()
plt.plot(range(len(test_cost_overtime)),test_cost_overtime,color='y',label='Original Model')
plt.legend()
plt.savefig('og Test Cost over time')

plt.figure()
plt.plot(range(len(test_acc_overtime)),test_acc_overtime,color='y',label='Original Model')
plt.legend()
plt.savefig('og Test Acc over time')



# -- end code --