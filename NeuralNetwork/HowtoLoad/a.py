# import tensorflow as tf
import numpy as np,sys,os
from numpy import float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize

np.random.seed(538)
# tf.set_random_seed(6785)


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)

    ** The names of the classes 
    0. aeroplane     [128,0,0]
    1. bicycle       [0,128,0]
    2. bird          [128,128,0]
    3. boat          [0,0,128]
    4. bottle        [128,0,128]
    5. bus           [0,128,128]
    6. car           [128,128,128]
    7. cat           [64,0,0]
    8. chair         [192,0,0]
    9. cow          [64,128,0]
    10. diningtable  [192,128,0]
    11. dog          [64,0,128]
    12. horse        [192,0,128]
    13. motorbike    [64,128,128]
    14. person       [192,128,128]
    15. potted plant [0,64,0]
    16. sheep        [128,64,0]
    17. sofa         [0,192,0]
    18. train        [128,192,0]
    19. tv/monitor   [0,64,128]
    20. Void/None    [0,0,0] or [224,224,192s]
    """
    return np.asarray([ [128,0,0], [0,128,0], [128,128,0],[0,0,128], [128,0,128], [0,128,128], [128,128,128],
                        [64,0,0], [192,0,0], [64,128,0], [192,128,0],[64,0,128], [192,0,128], [64,128,128], [192,128,128],
                        [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],[0,64,128]])

def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
            (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask



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
train_data,train_data_gt = shuffle(train_data,train_data_gt)

# create the array to read
train_images = np.zeros(shape=(50,128,128,3))
train_labels = np.zeros(shape=(50,128,128,3))
train_labels_channels = np.zeros(shape=(10,128,128,21))

for file_index in range(50):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='RGB'),(128,128))
    train_labels[file_index,:,:]   = imresize(imread(train_data_gt[file_index],mode='RGB'),(128,128))
train_images = train_images.astype(int)
train_labels = train_labels.astype(int)

print(len(get_pascal_labels()))
print('------------------------------')
test_label = np.expand_dims(train_labels[0,:,:,:],axis=0)

plt.imshow(np.squeeze(train_labels[0,:,:,:]) ) 
plt.show()
plt.imshow(np.squeeze(test_label[0,:,:,:]) ) 
plt.show()

mask_index = 0 
label_mask = np.zeros((128, 128,20)).astype(int)
for label in get_pascal_labels():
    coordinate = np.where(np.all(test_label == label, axis=-1))
    label_mask[coordinate[1],coordinate[2],mask_index ] = 1
    mask_index = mask_index + 1
    
for x in range(20):
    plt.imshow(label_mask[:,:,x],cmap='gray')
    plt.show()

# -- end code --