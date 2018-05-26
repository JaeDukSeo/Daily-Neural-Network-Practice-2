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

        with tf.Session(graph=self.graph) as session:
            self.u, self.v,self.singular_values, self.sigma = session.run([u,v, singular_values, sigma],feed_dict={self.X: self.data})


    def reduce(self, n_dimensions=None, keep_info=None):

        if keep_info:
            # Normalize singular values
            normalized_singular_values = self.singular_values / sum(self.singular_values)
            # Create the aggregated ladder of kept information per dimension
            ladder = np.cumsum(normalized_singular_values)
            # Get the first index which is above the given information threshold
            index = next(idx for idx, value in enumerate(ladder) if value >= keep_info) + 1
            n_dimensions = index
            print('N : ', n_dimensions)

        # Here is the reduction
        with self.graph.as_default():

            # Cut out the relevant part from sigma
            print(self.sigma.shape)
            sigma = tf.slice(self.sigma, [0, 0], [self.data.shape[1], n_dimensions])
            print(sigma.shape)
            print(self.u.shape)
            # PCA
            pca = tf.matmul(self.u,sigma)
            # pca = tf.matmul( tf.transpose(self.v),sigma)
            print(pca.shape)

        with tf.Session(graph=self.graph) as session:
            return session.run(pca, feed_dict={self.X: self.data})





def tf_pca(x):
    '''
        Compute PCA on the bottom two dimensions of x,
        eg assuming dims = [..., observations, features]
    '''
    # Center
    x -= tf.reduce_mean(x, -2, keepdims=True)

    # Currently, the GPU implementation of SVD is awful.
    # It is slower than moving data back to CPU to SVD there
    # https://github.com/tensorflow/tensorflow/issues/13222

    ss, us, vs = tf.svd(x, full_matrices=False, compute_uv=True)
    
    print(us.shape)
    print(ss.shape)
    print(vs.shape)
    print('-----')

    ss = tf.expand_dims(ss, -2)    
    print(ss.shape)
    projected_data = us * ss
    print(projected_data.shape)

    # Selection of sign of axes is arbitrary.
    # This replicates sklearn's PCA by duplicating flip_svd
    # https://github.com/scikit-learn/scikit-learn/blob/7ee8f97e94044e28d4ba5c0299e5544b4331fd22/sklearn/utils/extmath.py#L499
    r = projected_data
    abs_r = tf.abs(r)
    m = tf.equal(abs_r, tf.reduce_max(abs_r, axis=-2, keepdims=True))
    signs = tf.sign(tf.reduce_sum(r * tf.cast(m, r.dtype), axis=-2, keepdims=True))
    result = r * signs

    return result



# x = tf.placeholder(shape=[50,32,32,3],dtype=tf.float32)
# ss = tf.reshape(x,[50,-1])

# ff = tf_pca(ss)

# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())
    
#     s  = sess.run([ff],feed_dict={x:np.random.randn(50,32,32,3).astype(np.float32)})

#     print(s[0].shape)

# sys.exit()

iris_dataset = datasets.load_iris()
row = 1500
cal = 16 * 16 * 4
cal = 8*8*16   
cal = 4*4*64   
temp = np.random.randn(row,cal)
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
pca = tf_pca2.reduce(n_dimensions=cal//2)  # Results in 2 dimensions
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