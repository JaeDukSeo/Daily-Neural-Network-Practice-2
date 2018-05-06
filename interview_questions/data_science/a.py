import tensorflow as tf
import numpy as np
import sys, os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
from skimage.transform import resize
from skimage.feature import hog
from skimage import data, exposure
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.neighbors import KNeighborsClassifier
np.random.seed(514)
tf.set_random_seed(678)

# data
# mnist = input_data.read_data_sets('../../Dataset/MNIST/', one_hot=True)
# training_images, training_labels, testing_images, testing_labels = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# training_images = np.reshape(training_images,(-1,28,28,1))

# 1. feature vector
# example = imresize(np.squeeze(training_images[1,:,:,:]),(256,256))
# feature_descrip,hog_iamge = hog(example, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),visualise=True)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
# ax1.axis('off')
# ax1.imshow(example, cmap=plt.cm.gray)
# ax1.set_title("Feature Vector Size: "+str(len(feature_descrip)))
# ax2.axis('off')
# ax2.imshow(hog_iamge, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()

# example = imresize(np.squeeze(training_images[3,:,:,:]),(256,256))
# feature_descrip,hog_iamge = hog(example, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),visualise=True)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
# ax1.axis('off')
# ax1.imshow(example, cmap=plt.cm.gray)
# ax1.set_title("Feature Vector Size: "+str(len(feature_descrip)))
# ax2.axis('off')
# ax2.imshow(hog_iamge, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()

centers = [(3, 3), (5,15), (10, 5)]
X1, Y1 = make_blobs(n_features=2, centers=centers,n_samples=500,center_box=(-20,20),cluster_std=1.0 )
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X1, Y1) 
output = set(Y1)
print('How many Classes are there?: ',output)
new_user = np.array([8,15.5])
print('New User value of : ',new_user, ' K-NN Prediction :' ,neigh.predict(np.expand_dims(new_user,axis=0)) )
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.scatter(new_user[0], new_user[1], marker='o',c='red')
plt.title("Flavours of Ice Cream, Predicted for new user: " + str(neigh.predict(np.expand_dims(new_user,axis=0))))
plt.show()


# -- end code --
