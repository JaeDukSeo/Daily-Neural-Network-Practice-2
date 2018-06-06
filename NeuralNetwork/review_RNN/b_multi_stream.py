import tensorflow as tf,numpy as np,pandas as pd,os
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(678)
tf.set_random_seed(678)

def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32)  + ( tf_elu(tf.cast(tf.less_equal(x,0),tf.float32) * x) + 1.0)
def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1 - tf.square(tf_tanh(x))
def tf_soft(x): return tf.nn.softmax(x)

# Different noises of different channels
def gaussian_noise_layer(input_layer,std=1.0):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + 0.1*noise

def possin_layer(layer):
    noise = tf.random_poisson(lam=1.0,shape=tf.shape(layer),dtype=tf.float32)
    return 0.1*noise + layer

def uniform_layer(input_layer):
    noise = tf.random_uniform(shape=tf.shape(input_layer),minval=0.5,dtype=tf.float32)
    return 0.6*noise + input_layer

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

# class recurrent CNN and CNN
class RCNN():

    def __init__(self,timestamp,c_in,c_out,x_kernel,h_kernel,size):

        self.w = tf.Variable(tf.random_normal([x_kernel,x_kernel,c_in,c_out]))
        self.h = tf.Variable(tf.random_normal([h_kernel,h_kernel,c_out,c_out]))

        self.input_record   = tf.Variable(tf.zeros([timestamp,batch_size,size,size,c_in]))
        self.hidden_record  = tf.Variable(tf.zeros([timestamp+1,batch_size,size,size,c_out]))
        self.hiddenA_record = tf.Variable(tf.zeros([timestamp+1,batch_size,size,size,c_out]))
        
        self.m_x,self.v_x = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.m_h,self.v_h = tf.Variable(tf.zeros_like(self.h)),tf.Variable(tf.zeros_like(self.h))

    def feedfoward(self,input,timestamp):

        # assign the input for back prop
        hidden_assign = []
        hidden_assign.append(tf.assign(self.input_record[timestamp,:,:,:],input))

        # perform feed forward
        layer =  tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')  + \
        tf.nn.conv2d(self.hidden_record[timestamp,:,:,:,:],self.h,strides=[1,1,1,1],padding='SAME') 
        layerA = tf_elu(layer)

        # assign for back prop
        hidden_assign.append(tf.assign(self.hidden_record[timestamp+1,:,:,:],layer))
        hidden_assign.append(tf.assign(self.hiddenA_record[timestamp+1,:,:,:],layerA))

        return layerA, hidden_assign 

    def backprop(self,grad,timestamp):

        grad_1 = grad
        grad_2 = d_tf_elu(self.hidden_record[timestamp,:,:,:,:])
        grad_3_x = self.input_record[timestamp,:,:,:,:]
        grad_3_h = self.hiddenA_record[timestamp-1,:,:,:,:]

        grad_middle = grad_1 * grad_2

        grad_x = tf.nn.conv2d_backprop_filter(
            input=grad_3_x,filter_size = self.w.shape,
            out_backprop = grad_middle,strides=[1,1,1,1],padding='SAME'
        )

        grad_h = tf.nn.conv2d_backprop_filter(
            input=grad_3_h,filter_size = self.h.shape,
            out_backprop = grad_middle,strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_size = self.hiddenA_record[timestamp-1,:,:,:].shape,
            filter=self.h,out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        update_w = []
        # === update x ====
        update_w.append( tf.assign(self.m_x,beta_1*self.m_x + (1-beta_1) * grad_x)  )
        update_w.append( tf.assign(self.v_x,beta_2*self.v_x + (1-beta_2) * grad_x ** 2) )
        m_hat_x = self.m_x/(1-beta_1)
        v_hat_x = self.v_x/(1-beta_2)
        adam_middle_x = learning_rate/(tf.sqrt(v_hat_x) + adam_e)
        update_w.append( tf.assign(self.w_x, tf.subtract(self.w_x,adam_middle_x*m_hat_x))  )

        # === update h ====
        update_w.append( tf.assign(self.m_h,beta_1*self.m_h + (1-beta_1) * grad_h)  )
        update_w.append( tf.assign(self.v_h,beta_2*self.v_h + (1-beta_2) * grad_h ** 2) )
        m_hat_h = self.m_h/(1-beta_1)
        v_hat_h = self.v_h/(1-beta_2)
        adam_middle_h = learning_rate/(tf.sqrt(v_hat_h) + adam_e)
        update_w.append( tf.assign(self.w_h, tf.subtract(self.w_h,adam_middle_h*m_hat_h))  )
        
        return grad_pass,update_w
    
class FNN():

    def __init__(self,input_dim,hidden_dim):
        self.w = tf.Variable(tf.random_normal([input_dim,hidden_dim]))

    def feedforward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = tf_tanh(self.layer)
        return self.layerA

    def backprop(self,gradient=None):
        grad_part_1 = gradient
        grad_part_2 = d_tf_tanh(self.layer)
        grad_part_3 = self.input 

        grad_x_mid = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.matmul(tf.transpose(grad_part_3),grad_x_mid)
        grad_pass = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        update_w = []
        update_w.append(tf.assign(self.w, self.w - learning_rate * grad))
        return grad_pass,update_w

# PCA Layer following the implementation: https://ewanlee.github.io/2018/01/17/PCA-With-Tensorflow/
class PCA_Layer():
    
    def __init__(self,dim,channel):
        
        self.alpha = tf.Variable(tf.random_normal(shape=[dim-2,dim-2,channel],dtype=tf.float32))
        self.beta  = tf.Variable(tf.ones(shape=[channel],dtype=tf.float32))

        self.current_sigma = None
        self.moving_sigma = tf.Variable(tf.zeros(shape=[(dim*dim*channel),(dim*dim*channel)-108],dtype=tf.float32))

    def feedforward(self,input,is_training):
        update_sigma = []

        # 1. Get the input Shape and reshape the tensor into [Batch,Dim]
        width,channel = input.shape[1],input.shape[3]
        reshape_input = tf.reshape(input,[batch_size,-1])
        trans_input = reshape_input.shape[1]

        # 2. Perform SVD and get the sigma value and get the sigma value
        singular_values, u, _ = tf.svd(reshape_input,full_matrices=False)

        def training_fn(): 
            # 3. Training 
            sigma1 = tf.diag(singular_values)
            sigma = tf.slice(sigma1, [0,0], [trans_input, (width*width*channel)-108])
            pca = tf.matmul(u, sigma)
            update_sigma.append(tf.assign(self.moving_sigma,self.moving_sigma*0.9 + sigma* 0.1 ))
            return pca,update_sigma

        def testing_fn(): 
            # 4. Testing calculate hte pca using the Exponentially Weighted Moving Averages  
            pca = tf.matmul(u, self.moving_sigma)
            return pca,update_sigma

        pca,update_sigma = tf.cond(is_training, true_fn=training_fn, false_fn=testing_fn)
        pca_reshaped = tf.reshape(pca,[batch_size,(width-2),(width-2),channel])
        out_put = self.alpha * pca_reshaped +self.beta 
        
        return out_put,update_sigma

class CNN():
    
    def __init__(self,k,inc,out):
        self.w = tf.Variable(tf.random_normal([k,k,inc,out],stddev=0.05))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
        self.v_hat_prev = tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding=padding)
        self.layerA = tf_elu(self.layer)
        return self.layerA 

    def backprop(self,gradient,learing_rate_dynamic):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_elu(self.layer) 
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

        update_w = []
        
        update_w.append(
            tf.assign( self.m,self.m*beta1 + (1-beta1) * grad   )
        )

        v_t = self.v_prev *beta2 + (1-beta2) * grad ** 2 

        def f1(): return v_t
        def f2(): return self.v_hat_prev

        v_max = tf.cond(tf.greater(tf.reduce_sum(v_t), tf.reduce_sum(self.v_hat_prev) ) , true_fn=f1, false_fn=f2)
        adam_middel = learing_rate_dynamic/(tf.sqrt(v_max) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,self.m))))
        update_w.append(
            tf.assign( self.v_prev,v_t )
        )
        update_w.append(
            tf.assign( self.v_hat_prev,v_max )
        )        
        return grad_pass,update_w   

# data
mnist = input_data.read_data_sets("../../Dataset/MNIST/", one_hot=True)
train_batch,train_label = mnist.train.images,mnist.train.labels
test_batch ,test_label  = mnist.test.images,mnist.test.labels
train_batch = np.reshape(train_batch,[-1,28,28,1])
test_batch = np.reshape(test_batch,[-1,28,28,1])

print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

# hyper 
num_epoch = 21
batch_size = 50
print_size = 1
timestamp = 4

learning_rate = 0.0001
beta1,beta2,adam_e = 0.9,0.9,1e-8

# define class
input_stream0 = CNN(3,1,1)
input_stream1 = FNN(784,676)
input_stream3 = CNN(3,1,1)
input_stream4 = FNN(784,676)

l1 = RCNN(timestamp=timestamp,c_in=1,c_out=3,x_kernel=3,h_kernel=1,size=28)
l2 = CNN(3,3,10)
l3 = CNN(1,10,10)
l4 = CNN(1,10,10)

# graph 
x = tf.placeholder(shape=[batch_size,784],dtype=tf.float32)
y = tf.placeholder(shape=[batch_size,10],dtype=tf.float32)
phase = tf.placeholder(tf.bool)

x1 = input_stream0.feedforward(tf.reshape(x,[batch_size,28,28,1]),padding='VALID')
x2 = tf.reshape(input_stream1.feedforward(x),[batch_size,26,26,1])
x3 = input_stream3.feedforward(tf.reshape(x,[batch_size,28,28,1]),padding='VALID')
x4 = tf.reshape(input_stream4.feedforward(x),[batch_size,26,26,1])

x_inputs = [x1,x1,x2,x3]
layer1_rnn_up = []
for time in range(timestamp):
    layer_out,layer_up = l1.feedfoward(x_inputs[time],time)
    layer1_rnn_up.append(layer_up)

layer2 = l2.feedforward(layer_out)
layer3_Input = tf.nn.avg_pool(layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3 = l3.feedforward(layer3_Input)
layer4_Input = tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input)

final_global = tf.reduce_mean(layer4,[1,2])
final_soft = tf_soft(final_global)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_global,labels=y) )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)

        for batch_size_index in range(0,len(train_batch),batch_size):

            current_batch = train_batch[batch_size_index:batch_size_index+batch_size]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size]
            sess_result = sess.run([cost,accuracy,correct_prediction,auto_train,layer1_rnn_up],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter : ",iter, " current batch: ",batch_size_index, ' Current cost: ', sess_result[0],
            ' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]
            
        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size]
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size]
            sess_result = sess.run([cost,accuracy,correct_prediction,layer1_rnn_up],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],
            ' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n----------")
            print('Train Current cost: ', train_cota/(len(train_batch)/(batch_size)),' Current Acc: ', train_acca/(len(train_batch)/(batch_size) ),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/(batch_size)))
        train_cot.append(train_cota/(len(train_batch)/(batch_size)))
        test_acc.append(test_acca/(len(test_batch)/batch_size))
        test_cot.append(test_cota/(len(test_batch)/batch_size))
        test_cota,test_acca = 0,0
        train_cota,train_acca = 0,0

    # Normalize the cost of the training
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))
    test_cot = (test_cot-min(test_cot) ) / (max(test_cot)-min(test_cot))

    # training done now plot
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig("Case Train.png")

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("Case Test.png")


# -- end code --