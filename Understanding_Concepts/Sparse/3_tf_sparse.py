import numpy as np,sys 
import matplotlib.pylab as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')
# Principal component analysis
# Independent Component Analysis.
# LinearDiscriminantAnalysis
# t-distributed Stochastic Neighbor Embedding
from sklearn.decomposition import PCA,FastICA,FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

np.random.seed(678)
tf.set_random_seed(678)


# Make the Data set
def load_data_clear_cut():
    X,Y = make_classification(n_samples=100,n_features=3,
                              n_redundant=0,n_clusters_per_class=1,n_classes=3,class_sep=3.2,n_informative=3)
    return X,Y

def load_data_not_so_clear():
    X,Y = make_classification(n_samples=100,n_features=3,
                              n_redundant=0,n_clusters_per_class=1,n_classes=3,class_sep=0.2,n_informative=3)
    return X,Y

# class 
class sparse_filter():
    
    def __init__(self,outc,changec):
        self.w = tf.Variable(tf.random_normal([outc,changec],stddev=0.05))
        self.epsilon = 1e-8

    def getw(self): return self.w

    def soft_abs(self,value):
        return tf.sqrt(value ** 2 + self.epsilon)

    def feedforward(self,input):
        
        first  = tf.matmul(input,self.w)
        second = self.soft_abs(first)
        third  = tf.divide(second,tf.sqrt(tf.reduce_sum(second**2,axis=0)+self.epsilon))
        four = tf.divide(third,tf.sqrt(tf.reduce_sum(third**2,axis=1)[:,tf.newaxis] +self.epsilon))
        five = tf.reduce_sum(four)
        return five

# -------- clear cut difference in data ---------
# show the original data 
X,Y = load_data_clear_cut()
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o', c=Y)
plt.title('Original Data Shape : ' + str(X.shape))
plt.show()
plt.close('all')

# hyper
num_epoch = 1000
learning_rate = 0.01


# reduce the dim to 2
tf.reset_default_graph()
reduce_2 = tf.Graph()
with reduce_2.as_default():
    # define class - reduce to dim 2
    l_sparse = sparse_filter(3,2)
    trained_w = l_sparse.getw()

    input_value = tf.placeholder(shape=[100,3],dtype=tf.float32,name='input_value')

    layer1 = l_sparse.feedforward(input_value)

    cost = layer1
    auto_train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

with tf.Session(graph = reduce_2) as sess: 
    
    sess.run(tf.global_variables_initializer())

    # train
    for iter in range(num_epoch):
        sess_results = sess.run([cost,auto_train],feed_dict={input_value:X})
        print("Current Iter: ",iter, " Current cost: ",sess_results[0],end='\r')
        if iter % 100 == 0 : print('\n')

    # after all training is done plot the results
    trained_w = sess.run(trained_w)
    X_sparse = np.matmul(X,trained_w)
    plt.scatter(X_sparse[:, 0], X_sparse[:, 1], marker='o', c=Y, edgecolor='k')
    plt.title('Sprase Data Shape : ' + str(X_sparse.shape))
    plt.show()


print('\n------ MOVING TO THE NEXT GRAPH -------\n')


# reduce the dim to 1
tf.reset_default_graph()
reduce_1 = tf.Graph()
with reduce_1.as_default() as g:

    # define class - reduce to dim 1
    l_sparse = sparse_filter(3,1)
    trained_w = l_sparse.getw()

    input_value = tf.placeholder(shape=[100,3],dtype=tf.float32,name='input_value')

    layer1 = l_sparse.feedforward(input_value)

    cost = tf.reduce_sum(layer1,name='cost')
    auto_train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

with tf.Session(graph = reduce_1) as sess: 
    
    sess.run(tf.global_variables_initializer())

    # train
    for iter in range(num_epoch):
        sess_results = sess.run([cost,auto_train],feed_dict={input_value:X})
        print("Current Iter: ",iter, " Current cost: ",sess_results[0],end='\r')
        if iter % 100 == 0 : print('\n')

    # after all training is done plot the results
    trained_w = sess.run(trained_w)
    X_sparse = np.matmul(X,trained_w)
    plt.scatter(X_sparse[:, 0], [1] * len(X_sparse), marker='o', c=Y, edgecolor='k')
    plt.title('Sprase Data Shape : ' + str(X_sparse.shape))
    plt.show()






# -- end code --    