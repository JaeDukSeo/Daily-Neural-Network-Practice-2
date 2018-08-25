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
from scipy.ndimage import zoom
import seaborn as sns

np.random.seed(678)
np.set_printoptions(precision = 3,suppress =True)
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

# create layer
def np_sigmoid(x): return  1 / (1 + np.exp(-x))
def d_np_sigmoid(x): return np_sigmoid(x) * (1.0 - np_sigmoid(x))

def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    return a

def grads(X, Y, weights):
    grads = np.empty_like(weights)
    a = feed_forward(X, weights)

    # ==== REFENCE ====
    # delta = (a[-1] - Y)
    # temp = a[-2].dot(weights[-1])
    # delta = delta * 1.0*(temp>0.0)
    # grads[-1] = a[-2].T.dot(delta)
    #
    # delta = (a[1] > 0.) * delta.dot(weights[1].T)
    # grads[0] = a[0].T.dot(delta)
    # for i in range(len(a)-2, 0, -1):
    #     delta = (a[i] > 0.) * delta.dot(weights[i].T)
    #     grads[i-1] = a[i-1].T.dot(delta)
    # ==== REFENCE ====

    # ==== MINE ====
    delta = a[2] - Y
    grad_1_2 = a[2] - Y
    grad_2_2 = 1.0*(a[1].dot(weights[1])>0.)
    grad_3_2 = a[1]
    grads[1] = grad_3_2.T.dot(grad_1_2 * grad_2_2) / batch_size

    grad_1_1 = (grad_1_2 * grad_2_2).dot(weights[1].T)
    grad_2_1 = 1.0*(a[0].dot(weights[0])>0.)
    grad_3_1 = a[0]
    grads[0] = grad_3_1.T.dot(grad_1_1 * grad_2_1) / batch_size
    # ==== MINE ====

    return grads

# import data
# mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
mnist = input_data.read_data_sets('../../Dataset/fashionmnist/',one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# trX, trY, teX, teY = mnist.load_data()

# weights = [np.random.normal(*w) * 0.1 for w in [(784, 100),(100,10) ]]
r = np.random.RandomState(1234)
weights = [
r.normal(0,0.05,size=(784, 100)),
r.normal(0,0.05,size=(100, 10)),
]

num_epochs, batch_size, learn_rate = 30, 20, 0.02

for i in range(num_epochs):

    for j in range(0, len(trX), batch_size):
        X = trX[j:j+batch_size]
        Y = trY[j:j+batch_size]
        weights = weights - learn_rate * grads(X, Y, weights)

        # gradd = grads(X, Y, weights)
        # m = m*0.9 + (1-0.9) * gradd
        # v = v*0.999 + (1-0.999) * gradd ** 2
        # m_hat,v_hat = m/(1-0.9),v/(1-0.999)
        # weights = weights - learn_rate * m_hat / ((v_hat) ** 0.5 + 10e-8)

    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    print(i, np.mean(prediction == np.argmax(teY, axis=1)))
    sys.exit()

for xxx in range(10):
    print('====================')
    print(teY[xxx,:])
    temp = feed_forward(teX[xxx,:], weights)[-1]
    print(temp)
    print('====================')
sys.exit()

# -- end code ---
