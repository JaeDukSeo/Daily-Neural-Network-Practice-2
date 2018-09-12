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

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

# data
data_location = "../../Dataset/STL10/img/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".png" in filename.lower() :
            train_data.append(os.path.join(dirName,filename))

image_resize_px = 96
train_images = np.zeros(shape=(len(train_data),image_resize_px,image_resize_px,1))

for file_index in range(len(train_images)):
    train_images[file_index,:,:]   = np.expand_dims(imresize(imread(train_data[file_index],mode='L'),(image_resize_px,image_resize_px)),2)

# normalize
train_batch= train_images/255.0

# print out the data shape and the max and min value
print('Train batch, min, max : ',train_batch.shape,train_batch.min(),train_batch.max())
# -- end code --
