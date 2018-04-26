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

# reshape data
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))

# rotate data
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)

# Normalize data from 0 to 1 per each channel
train_batch[:,:,:,0]  = (train_batch[:,:,:,0] - train_batch[:,:,:,0].min(axis=0)) / (train_batch[:,:,:,0].max(axis=0) - train_batch[:,:,:,0].min(axis=0))
train_batch[:,:,:,1]  = (train_batch[:,:,:,1] - train_batch[:,:,:,1].min(axis=0)) / (train_batch[:,:,:,1].max(axis=0) - train_batch[:,:,:,1].min(axis=0))
train_batch[:,:,:,2]  = (train_batch[:,:,:,2] - train_batch[:,:,:,2].min(axis=0)) / (train_batch[:,:,:,2].max(axis=0) - train_batch[:,:,:,2].min(axis=0))
test_batch[:,:,:,0]  = (test_batch[:,:,:,0] - test_batch[:,:,:,0].min(axis=0)) / (test_batch[:,:,:,0].max(axis=0) - test_batch[:,:,:,0].min(axis=0))


# class
class cnn():
    
    def __init__(self,k,inc,out):
        self.w1 = tf.Variable(tf.random_normal([k,k,inc,out]))
        self.w2 = tf.Variable(tf.random_normal([k,k,out,out]))
        
        self.m,self.v = tf.Variable(tf.zeros_like(self.w1)),tf.Variable(tf.zeros_like(self.w1))

    def feedforward(self,input,resadd=True):
        self.input  = input

        self.layer1  = tf.nn.batch_normalization(self.input ,mean=0,variance=1.0,variance_epsilon=1e-8,offset=True,scale=True)
        self.layer1  = tf_relu(self.layer1) 
        self.layer1  = tf.nn.conv2d(self.layer1,self.w1,strides=[1,1,1,1],padding='SAME')

        self.layer1  = tf.nn.batch_normalization(self.layer1 ,mean=0,variance=1.0,variance_epsilon=1e-8,offset=True,scale=True)
        self.layer1  = tf_relu(self.layer1) 
        self.layer1  = tf.nn.conv2d(self.layer1,self.w2,strides=[1,1,1,1],padding='SAME')

        if resadd: return self.layer1 + self.input
        return self.layer1

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
learning_rate = 0.0001
beta1,beta2,adame = 0.9,0.999,1e-8







# class
l1 = cnn(3,3,16)

l2_1 = cnn(3,16,32)
l2_2 = cnn(3,32,32)
l2_3 = cnn(3,32,32)
l2_4 = cnn(3,32,32)

l3_1 = cnn(3,32,64)
l3_2 = cnn(3,64,64)
l3_3 = cnn(3,64,64)
l3_4 = cnn(3,64,64)

l4_1 = cnn(3,64,128)
l4_2 = cnn(3,128,128)
l4_3 = cnn(3,128,128)
l4_4 = cnn(3,128,10)








# graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

layer1 = l1.feedforward(x,resadd=False)

layer2Input = tf.nn.avg_pool(layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2_1 = l2_1.feedforward(layer2Input,resadd=False)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_3 = l2_3.feedforward(layer2_2)
layer2_4 = l2_4.feedforward(layer2_3)

layer3Input = tf.nn.avg_pool(layer2_4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3_1 = l3_1.feedforward(layer3Input,resadd=False)
layer3_2 = l3_2.feedforward(layer3_1)
layer3Input2 = tf.nn.avg_pool(layer3_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3_3 = l3_3.feedforward(layer3Input2)
layer3_4 = l3_4.feedforward(layer3_3)

layer4Input = tf.nn.avg_pool(layer3_4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4_1 = l4_1.feedforward(layer4Input,resadd=False)
layer4_2 = l4_2.feedforward(layer4_1)
layer4Input2 = tf.nn.avg_pool(layer4_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
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