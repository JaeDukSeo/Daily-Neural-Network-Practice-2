# import tensorflow as tf
import numpy as np,sys,os
from numpy import float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize

np.random.seed(6783)
# tf.set_random_seed(6785)

# data 
data_location =  "../../Dataset/VOC2011/SegmentationClass/"
train_data_gt = []  # create an empty list
only_file_name = []
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".png" in filename.lower() :
            train_data_gt.append(os.path.join(dirName,filename))
            only_file_name.append(filename[:-4])

data_location = "../../Dataset/VOC2011/JPEGImages/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".jpg" in filename.lower() and filename.lower()[:-4] in  only_file_name:
            train_data.append(os.path.join(dirName,filename))

train_images = np.zeros(shape=(10,128,128,3))
train_labels = np.zeros(shape=(10,128,128,3))

for file_index in range(10):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='RGB'),(128,128)).astype(int)
    train_labels[file_index,:,:]   = imresize(imread(train_data_gt[file_index],mode='RGB'),(128,128)).astype(int)

for x in range(10):
    plt.imshow(train_images[x,:,:,:])
    plt.show()
    plt.imshow(train_labels[x,:,:,:])
    plt.show()


train_labels_channels = np.zeros(shape=(10,128,128,20))



# -- end code --