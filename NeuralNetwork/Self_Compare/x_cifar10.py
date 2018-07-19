#For plotting the images
from matplotlib import pyplot as plt
import sys
import numpy as np
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
from skimage.color import rgba2rgb

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(6278)
tf.set_random_seed(6728)
ia.seed(6278)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# SOM as layer
class SOM_Layer(): 

    def __init__(self,m,n,dim,learning_rate_som = 0.04,radius_factor = 1.1,gaussian_std=0.5):
        
        self.m = m
        self.n = n
        self.dim = dim
        self.gaussian_std = gaussian_std
        self.map = tf.Variable(tf.random_normal(shape=[m*n,dim],stddev=0.05))
        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
        self.alpha = learning_rate_som
        self.sigma = max(m,n)*1.1

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def getmap(self): return self.map
    def getlocation(self): return self.bmu_locs

    def feedforward(self,input):
    
        self.input = input
        self.squared_distance = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.map, axis=0),tf.expand_dims(self.input, axis=1)), 2), 2)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        self.bmu_locs = tf.reshape(tf.gather(self.location_vects, self.bmu_indices), [-1, 2])

    def backprop(self,iter,num_epoch):

        # Update the weigths 
        radius = tf.subtract(self.sigma,
                                tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                    tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        alpha = tf.subtract(self.alpha,
                            tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                      tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        self.bmu_distance_squares = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2), 
            2)

        self.neighbourhood_func = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, "float32")), tf.multiply(
                tf.square(tf.multiply(radius, self.gaussian_std)), 2)))

        self.learning_rate_op = tf.multiply(self.neighbourhood_func, alpha)
        
        self.numerator = tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.learning_rate_op, axis=-1),
            tf.expand_dims(self.input, axis=1)), axis=0)

        self.denominator = tf.expand_dims(
            tf.reduce_sum(self.learning_rate_op,axis=0) + float(1e-20), axis=-1)

        self.new_weights = tf.div(self.numerator, self.denominator)
        self.update = tf.assign(self.map, self.new_weights)

        return self.update

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

# hyper parameter 10000
num_epoch = 100
batch_size = 100

learning_rate = 0.000000001
beta1,beta2,adam_e = 0.9,0.999,1e-8

# print out the data shape
train_batch = train_batch[:500,:,:,:]
train_label = train_label[:500,:]
test_batch = test_batch[:batch_size,:,:,:]
test_label = test_label[:batch_size,:]

print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

train_batch = train_batch/255.0
test_batch = test_batch/255.0
train_batch = np.reshape(train_batch,[500,-1])
test_batch = np.reshape(test_batch,[batch_size,-1])

# hyper parameter
map_width_height  = 30
map_dim = 32 * 32 * 3
num_epoch = 100
batch_size = 100

# class
SOM_layer = SOM_Layer(map_width_height,map_width_height,map_dim,learning_rate_som=0.8,radius_factor=0.4,gaussian_std = 0.03)

# create the graph
x = tf.placeholder(shape=[None,map_dim],dtype=tf.float32)
current_iter = tf.placeholder(shape=[],dtype=tf.float32)

# graph
SOM_layer.feedforward(x)
map_update=SOM_layer.backprop(current_iter,num_epoch)

# session
with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    # start the training
    for iter in range(num_epoch):
        for current_train_index in range(0,len(test_batch),batch_size):
            currren_train = train_batch[current_train_index:current_train_index+batch_size]
            sess_results = sess.run(map_update,feed_dict={x:currren_train,current_iter:iter})
            print('Current Iter: ',iter,' Current Train Index: ',current_train_index,' Current SUM of updated Values: ',sess_results.sum(),end='\r' )
        print('\n-----------------------')

    # after training is done get the cloest vector
    locations = sess.run(SOM_layer.getlocation(),feed_dict={x:train_batch[:batch_size]})
    x1 = locations[:,0]; y1 = locations[:,1]
    index = [ np.where(r==1)[0][0] for r in train_label ]
    index = list(map(str, index))

    ## Plots: 1) Train 2) Test+Train ###
    plt.figure(1, figsize=(12,6))
    plt.subplot(121)
    plt.scatter(x1,y1)
    # Just adding text
    for i, m in enumerate(locations):
        plt.text( m[0], m[1],index[i], ha='center', va='center', 
        bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.title('Train CIFAR 100')

    locations2 = sess.run(SOM_layer.getlocation(),feed_dict={x:test_batch})
    locations2 = locations2[:50]
    x2 = locations2[:,0]; y2 = locations2[:,1]
    index2 = [ np.where(r==1)[0][0] for r in test_label ]
    index2 = list(map(str, index2))

    plt.subplot(122)
    # Plot 2: Training + Testing
    plt.scatter(x1,y1)
    # Just adding text
    for i, m in enumerate(locations):
        plt.text( m[0], m[1],index[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.scatter(x2,y2)
    # Just adding text
    for i, m in enumerate(locations2):
        plt.text( m[0], m[1],index2[i], ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5, lw=0))
    plt.title('Test CIFAR 50 + Train CIFAR 100')
    plt.show()


# -- end code --