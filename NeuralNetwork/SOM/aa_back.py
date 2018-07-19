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

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(6278)
tf.set_random_seed(6728)
from som import SOM

class SOM_Layer(): 

    def __init__(self,m,n,dim):
        
        self.m = m
        self.n = n
        self.dim = dim
        # self.map = tf.Variable(tf.random_normal(shape=[m*n,dim]))
        self.map = tf.Variable(tf.random_uniform(shape=[m*n,dim],minval=0,maxval=1,seed=2))
        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
        self.alpha = 0.03
        self.sigma = max(m, n) / 2.0

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def getw(self): return self.map

    def feedforward(self,input):
    
        self.input = input
        self.squared_distance = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.map, axis=0),tf.expand_dims(self.input, axis=1)), 2), 2)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        self.bmu_locs = tf.reshape(tf.gather(self.location_vects, self.bmu_indices), [-1, 2])

    def backprop(self,iter,num_epoch):
        
        # With each epoch, the initial sigma value decreases linearly
        radius = tf.subtract(self.sigma,tf.multiply(iter,
                                             tf.divide(tf.cast(tf.subtract(self.sigma, 1),tf.float32),
                                                       tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        alpha = tf.subtract(self.alpha,
                                tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                      tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        bmu_distance_squares = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2), 
            2)

        neighbourhood_func = tf.exp(tf.divide(tf.negative(tf.cast(
                bmu_distance_squares, "float32")), tf.multiply(
                tf.square(tf.multiply(radius, 0.3)), 2)))

        learning_rate_op = tf.multiply(neighbourhood_func, alpha)
        numerator = tf.reduce_sum(tf.multiply(tf.expand_dims(learning_rate_op, axis=-1),tf.expand_dims(self.input, axis=1)), axis=0)
        denominator = tf.expand_dims(tf.reduce_sum(learning_rate_op,axis=0) + float(1e-12), axis=-1)
        new_weights = tf.divide(numerator, denominator)
        update = tf.assign(self.map,new_weights)  

        return update

#Training inputs for RGBcolors
colors = np.array([[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])

color_names = ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

SOM_layer = SOM_Layer(5,5,3)

num_epoch = 500
batch_size = 1

x = tf.placeholder(shape=[batch_size,3],dtype=tf.float32)
current_iter = tf.placeholder(shape=[],dtype=tf.float32)

SOM_layer.feedforward(x)
map_update=SOM_layer.backprop(current_iter,num_epoch)


with tf.Session() as sess: 

      sess.run(tf.global_variables_initializer())

      for iter in range(num_epoch):
            for current_train_index in range(0,len(colors),batch_size):
                  currren_train = colors[current_train_index:current_train_index+batch_size]
                  sess.run(map_update,feed_dict={x:currren_train,current_iter:iter})

      som_map = sess.run(SOM_layer.getw())
      print(som_map)
      plt.imshow(som_map.reshape(5,5,3))
      plt.title('Color SOM')
      plt.show()

sys.exit()
# Train a 20x30 SOM with 400 iterations
som = SOM(5, 5, 3, 400)

# plot the weight maps
image_grid = som.get_centroids()
plt.imshow(image_grid)
plt.title('Color SOM')
plt.show()


# -- end code --