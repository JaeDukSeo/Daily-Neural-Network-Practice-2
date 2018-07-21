import numpy as np
import sys
import tensorflow as tf
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread,imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from imgaug import augmenters as iaa
import imgaug as ia
from skimage.color import rgba2rgb
np.set_printoptions(precision=2)

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

AA = tf.placeholder(shape=[3,2],dtype=tf.float32)

def tf_neg_squared_euc_dists(A):
    r = tf.reduce_sum(A*A, 1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return -D 

def tf_softmax_with_max_trick(A):
    two_max = tf.reshape(tf.reduce_max(A,1),[-1,1])
    two_exp = tf.exp(A-two_max)
    two_zero = tf.matrix_set_diag(two_exp,tf.zeros([two_exp.shape[0].value],dtype=tf.float32)) \
    +  tf.constant(1e-8)
    tf_soft = two_zero/tf.reshape(tf.reduce_sum(two_zero,1),[-1,1])
    return tf_soft

temp = np.array([
    [1,1],
    [2,2],
    [1,1]
]).astype(np.float32)

sess = tf.Session()
print(sess.run(tf_neg_squared_euc_dists(temp)))
temp = sess.run(tf_neg_squared_euc_dists(temp))
print(np.around(sess.run(tf_softmax_with_max_trick(temp)),2))
temp = sess.run(tf_softmax_with_max_trick(temp))






# -- end code ---