import tensorflow as tf
import numpy as np,sys
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(789)
tf.set_random_seed(789)

class TF_PCA():
    
    def __init__(self,data,label):
        
        self.data = data
        self.dtype = tf.float32
        self.target = label         

    def fit(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(self.dtype, shape=self.data.shape)
            # Perform SVD
            singular_values, u, v = tf.svd(self.X,full_matrices=False)
            # Create sigma matrix
            sigma = tf.diag(singular_values)

            print('singular_values:',singular_values.shape)
            print('u:',u.shape)
            print('v:',v.shape)
            print('sigma:',sigma.shape)

            # sigma1 = tf.slice(sigma, [0, 0], [self.data.shape[1], 512])
            # print('-------f1------')
            # print(u.shape)
            # print(sigma1.shape)
            # print('-------f1------')
            # pca = tf.matmul(u,sigma1)

        with tf.Session(graph=self.graph) as session:
            self.u, self.v,self.singular_values, self.sigma = session.run([u,v, singular_values, sigma],feed_dict={self.X: self.data})


    def reduce(self, n_dimensions=None, keep_info=None):

        with self.graph.as_default():
            # Cut out the relevant part from sigma
            sigma = tf.slice(self.sigma, [0, 0], [self.data.shape[1], n_dimensions])
            # PCA
            pca = tf.matmul(self.u, sigma)
        with tf.Session(graph=self.graph) as session:
            return session.run(pca, feed_dict={self.X: self.data})






# x = tf.placeholder(shape=[50,32,32,3],dtype=tf.float32)
# ss = tf.reshape(x,[50,-1])

# ff = tf_pca(ss)

# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())
    
#     s  = sess.run([ff],feed_dict={x:np.random.randn(50,32,32,3).astype(np.float32)})

#     print(s[0].shape)

# sys.exit()

iris_dataset = datasets.load_iris()
row = 250
# cal = 16 * 16 * 4
cal = 8 * 8 * 3
# cal = 8*8*16   
# cal = 4*4*64   
temp = np.random.randn(row,cal)
# temp = np.random.randn(cal,row)
#  8 8 4 = 256
#  4 4 4 = 64
temps = np.ones((row,) )
print(iris_dataset.data.shape)
print(iris_dataset.target.shape)
print(temp.shape)
print(temps.shape)




print('------------------------------------')

tf_pca2 = TF_PCA(temp, temps)
# tf_pca2 = TF_PCA(iris_dataset.data, iris_dataset.target)
tf_pca2.fit()
pca = tf_pca2.reduce(n_dimensions=cal//4)  # Results in 2 dimensions

print(temp.shape)
print(pca.shape)

sys.exit()

print('------------------------------------')
print('------------------------------------')
print('------------------------------------')
print('------------------------------------')
print('------------------------------------')
print('------------------------------------')
tf_pca = TF_PCA(temp, temps)
tf_pca.fit()
pca = tf_pca.reduce(n_dimensions=64)  # Results in 2 dimensions
sys.exit()

pca = tf_pca.reduce(keep_info=0.5)  # Results in 2 dimensions

sys.exit()

color_mapping = {0: sns.xkcd_rgb['bright purple'], 1: sns.xkcd_rgb['lime'], 2: sns.xkcd_rgb['ochre']}
colors = list(map(lambda x: color_mapping[x], tf_pca.target))
plt.scatter(pca[:, 0], pca[:, 1], c=colors)
plt.show()






# -- end code --