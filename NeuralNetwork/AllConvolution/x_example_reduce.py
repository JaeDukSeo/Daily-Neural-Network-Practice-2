import tensorflow as tf
import numpy as np
import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

tf.set_random_seed(678)
np.random.seed(678)

matrix = np.random.randn(4,4).astype(np.float32)
print(matrix)
matrix_expand = np.expand_dims(matrix,axis=0)
matrix_expand = np.expand_dims(matrix_expand,axis=3)
print(matrix.shape)
print(matrix_expand.shape)

# reduce mean
x = tf.placeholder(shape=[1,4,4,1],dtype=tf.float32)
result = tf.reduce_mean(x,[1,2])

# Intialize the Session
sess = tf.Session()

# Print the result
print('----------------------')
print(matrix)
print("Numpy Result: ",matrix.mean())
print("TF Result: ", sess.run(result,feed_dict={x:matrix_expand}))
print('---------------------')

# Close the session
sess.close()


# -- end code -- 