import tensorflow as tf
import numpy as np,sys,os
from numpy import float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize

np.random.seed(6783)
tf.set_random_seed(6785)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),tf.float32)
def tf_LRelu(x): return tf.nn.leaky_relu(x)
def d_tf_LRelu(x): return tf.cast(tf.greater(x,0),tf.float32) + tf.cast(tf.less_equal(x,0),tf.float32) *x* 0.2
def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))
def tf_log(x): return tf.nn.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1-tf_log(x))
def tf_softmax(x): return tf.nn.softmax(x)

# convolution layer
class CNNLayer():
    
    def __init__(self,ker,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.001))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return [self.w]

    def feedforward(self,input,stride=1,padding_val='SAME',batch_norm=True,mean_pooling=True,no_activation=False):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding=padding_val)
        if batch_norm: self.layer = tf.nn.batch_normalization(self.layer,mean=0.0,variance=1.0,variance_epsilon=1e-8,scale=True,offset=True)
        if no_activation: return self.layer
        self.layerA = self.act(self.layer)
        if mean_pooling: self.layerA = tf.nn.avg_pool(self.layerA,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
        return self.layerA

    def backprop(self,gradient,stride=1):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,filter_sizes = self.w.shape,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        grad_pass  = tf.nn.conv2d_backprop_input(
            input_sizes=[batch_size] + list(self.input.shape[1:]),filter = self.w ,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        update_w = []

        update_w.append(
            tf.assign( self.m,self.m*beta_1 + (1-beta_1) * grad   )
        )
        update_w.append(
            tf.assign( self.v,self.v*beta_2 + (1-beta_2) * grad ** 2   )
        )

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,update_w

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor

# Def: Get Pascal Pixel Labels for the dataset
def get_pascal_labels_pixel():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)

    ** The names of the classes 
    0. Void/None    [0,0,0] or [224,224,192s]
    1. aeroplane     [128,0,0]
    2. bicycle       [0,128,0]
    3. bird          [128,128,0]
    4. boat          [0,0,128]
    5. bottle        [128,0,128]
    6. bus           [0,128,128]
    7. car           [128,128,128]
    8. cat           [64,0,0]
    9. chair         [192,0,0]
    10. cow          [64,128,0]
    11. diningtable  [192,128,0]
    12. dog          [64,0,128]
    13. horse        [192,0,128]
    14. motorbike    [64,128,128]
    15. person       [192,128,128]
    16. potted plant [0,64,0]
    17. sheep        [128,64,0]
    18. sofa         [0,192,0]
    19. train        [128,192,0]
    20. tv/monitor   [0,64,128]
    """
    return np.asarray([ [128,0,0], [0,128,0], [128,128,0],[0,0,128], [128,0,128], [0,128,128], [128,128,128],
                        [64,0,0], [192,0,0], [64,128,0], [192,128,0],[64,0,128], [192,0,128], [64,128,128], [192,128,128],
                        [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],[0,64,128]])

# Def: Get Pascal Name Labels for the dataset
def get_pascal_label_names():
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
    return np.asarray([ ['aeroplane'],['bicycle'],['bird'],['boat'],['bottle'],['bus'],['car'],['cat']
                       ,['chair'],['cow'],['diningtable'],['dog'],['horse'],['motorbike'],['person'],['potted plant'],
                       ['sheep'],['sofa'],['train'],['tv/monitor'] ])

# Def: Encode the Segmentation Mask into each class
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
    for ii, label in enumerate(get_pascal_labels_pixel()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

# Def: Decode the Segmentation Mask into each class
def decode_segmap(mask,num_class=20):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at a given location is the integer denoting the class index.
    """
    mask = mask > 0.1
    mask_index = 0 
    label_mask = np.zeros((mask.shape[0], mask.shape[1],3)).astype(int)
    color_label_list = get_pascal_labels_pixel()
    for class_index in range(num_class):
        coordinate = np.where(mask[:,:,class_index] == 1)
        label_mask[coordinate[0],coordinate[1],:] = color_label_list[class_index]
        mask_index = mask_index + 1
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

NUMBER_OF_IMAGE = 300
train_images = np.zeros(shape=(NUMBER_OF_IMAGE,128,128,3))
train_labels_RGB = np.zeros(shape=(NUMBER_OF_IMAGE,128,128,3))
train_labels = np.zeros(shape=(NUMBER_OF_IMAGE,128,128,1))

for file_index in range(len(train_images)):
    train_images[file_index,:,:,:]     = imresize(imread(train_data[file_index],mode='RGB'),(128,128))
    train_labels_RGB[file_index,:,:,:] = imresize(imread(train_data_gt[file_index],mode='RGB'),(128,128))
    train_labels[file_index,:,:,:]     = np.expand_dims(encode_segmap(train_labels_RGB[file_index,:,:,:]),axis=3)

train_images = train_images.astype(np.float32)
train_labels = train_labels.astype(np.float32)
train_labels_RGB = train_labels_RGB.astype(int)

train_images[:,:,:,0]  = (train_images[:,:,:,0] - train_images[:,:,:,0].min(axis=0)) / (train_images[:,:,:,0].max(axis=0) - train_images[:,:,:,0].min(axis=0)+1e-10)
train_images[:,:,:,1]  = (train_images[:,:,:,1] - train_images[:,:,:,1].min(axis=0)) / (train_images[:,:,:,1].max(axis=0) - train_images[:,:,:,1].min(axis=0)+1e-10)
train_images[:,:,:,2]  = (train_images[:,:,:,2] - train_images[:,:,:,2].min(axis=0)) / (train_images[:,:,:,2].max(axis=0) - train_images[:,:,:,2].min(axis=0)+1e-10)

# train_labels[:,:,:,0]  = (train_labels[:,:,:,0] - train_labels[:,:,:,0].min(axis=0)) / (train_labels[:,:,:,0].max(axis=0) - train_labels[:,:,:,0].min(axis=0)+1e-10)
# train_labels[:,:,:,1]  = (train_labels[:,:,:,1] - train_labels[:,:,:,1].min(axis=0)) / (train_labels[:,:,:,1].max(axis=0) - train_labels[:,:,:,1].min(axis=0)+1e-10)
# train_labels[:,:,:,2]  = (train_labels[:,:,:,2] - train_labels[:,:,:,2].min(axis=0)) / (train_labels[:,:,:,2].max(axis=0) - train_labels[:,:,:,2].min(axis=0)+1e-10)

test_image = train_images[290:,:,:,:]
test_label = train_labels[290:,:,:,:]
train_labels_RGB_test = train_labels_RGB[290:,:,:,:]

train_image = train_images[:290,:,:,:]
train_label = train_labels[:290,:,:,:]
train_labels_RGB = train_labels_RGB[:290,:,:,:]
# hyper
NUM_CLASS = 20
num_epoch = 1000
learing_rate = 0.00001
batch_size = 1
print_size = 10

# define 
l1_e = CNNLayer(3,3,32,tf_Relu,d_tf_Relu)
l2_e = CNNLayer(3,32,64,tf_Relu,d_tf_Relu)
l3_e = CNNLayer(3,64,128,tf_Relu,d_tf_Relu)
l4_e = CNNLayer(3,128,256,tf_Relu,d_tf_Relu)
l5_e = CNNLayer(3,256,256,tf_Relu,d_tf_Relu)

l6_d = CNNLayer(3,256,256,tf_Relu,d_tf_Relu)
l7_d = CNNLayer(3,256,128,tf_Relu,d_tf_Relu)
l8_d = CNNLayer(3,128,64,tf_Relu,d_tf_Relu)
l9_d = CNNLayer(3,64,32,tf_Relu,d_tf_Relu)
l10_d = CNNLayer(3,32,20,tf_Relu,d_tf_Relu)

# graph
x = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,128,128,20],dtype=tf.float32)

layer1 = l1_e.feedforward(x)
layer2 = l2_e.feedforward(layer1)
layer3 = l3_e.feedforward(layer2)
layer4 = l4_e.feedforward(layer3)
layer5 = l5_e.feedforward(layer4)

layer6_Input = tf_repeat(layer5,[1,2,2,1])
layer6 = l6_d.feedforward(layer6_Input,mean_pooling=False,batch_norm=False)

layer7_Input = tf_repeat(layer6,[1,2,2,1])
layer7 = l7_d.feedforward(layer7_Input,mean_pooling=False,batch_norm=False)

layer8_Input = tf_repeat(layer7,[1,2,2,1])
layer8 = l8_d.feedforward(layer8_Input,mean_pooling=False,batch_norm=False)

layer9_Input = tf_repeat(layer8,[1,2,2,1])
layer9 = l9_d.feedforward(layer9_Input,mean_pooling=False,batch_norm=False)

layer10_Input = tf_repeat(layer9,[1,2,2,1])
layer10 = l10_d.feedforward(layer10_Input,mean_pooling=False,batch_norm=False)

layer10_reshape = tf.reshape(layer10,[-1,NUM_CLASS])
y_reshape = tf.reshape(y,[-1,NUM_CLASS])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer10_reshape,labels=y_reshape))
auto_train = tf.train.MomentumOptimizer(learning_rate=learing_rate,momentum=0.9).minimize(cost)

# layer10_predict = tf.reshape(tf_softmax(layer10_reshape),[batch_size,128,128,20])
layer10_predict = tf.argmax(layer10,axis=3)


# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        # train_image,train_label,train_labels_RGB = shuffle(train_image,train_label,train_labels_RGB)
        for current_batch_index in range(0,len(train_image),batch_size):
            
            current_image_batch = train_image[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_mask_batch  = train_label[current_batch_index:current_batch_index+batch_size,:,:,:]

            sess_results = sess.run([auto_train,cost],feed_dict={x:current_image_batch,y:current_mask_batch})
            print("Current Iter: ",iter, " current batch: ",current_batch_index,' Cost : ',sess_results[1],end='\r')

        if iter % print_size == 0:
            print("\n------------------------\n")
            test_example    = train_image[:batch_size,:,:,:]
            test_example_gt_RGB = train_labels_RGB[:batch_size,:,:,:]
            sess_results = sess.run([layer10_predict],feed_dict={x:test_example})

            sess_results =  sess_results[0][:,:,:]
            print(sess_results.shape)
            test_example =    test_example[0,:,:,:]
            test_example_gt = test_example_gt_RGB[0,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example))
            plt.axis('off')
            plt.title('Original Image')
            plt.savefig('train_change/'+str(iter)+"a Original Image.png")

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt))
            plt.axis('off')
            plt.title('Ground Truth Mask')
            plt.savefig('train_change/'+str(iter)+"b Original Mask.png")

            plt.figure()
            plt.axis('off')
            plt.imshow(np.squeeze(sess_results[:,:,:]),cmap='gray')
            plt.title("Generated Mask")
            plt.savefig('train_change/'+str(iter)+"c Generated Mask.png")

            plt.close('all')       

    # Print halve test
    # for current_batch_index in range(0,len(test_images),batch_size):
    #     test_example = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     test_example_gt  = test_images_c[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     sess_results = sess.run([g_e_layer_final],feed_dict={input_binary_image:test_example})

    #     sess_results = sess_results[0][0,:,:,:]
    #     test_example = test_example[0,:,:,:]
    #     test_example_gt = test_example_gt[0,:,:,:]

    #     plt.figure()
    #     plt.imshow(np.squeeze(test_example),cmap='gray')
    #     plt.axis('off')
    #     plt.title('Original Mask ')
    #     plt.savefig('gif/'+str(current_batch_index)+"a_Original_Mask.png")

    #     plt.figure()
    #     plt.imshow(np.squeeze(test_example_gt))
    #     plt.axis('off')
    #     plt.title('Ground Truth Image')
    #     plt.savefig('gif/'+str(current_batch_index)+"b_Original_Image.png")

    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(np.squeeze(sess_results)   ,cmap='gray')
    #     plt.title("Generated Image")
    #     plt.savefig('gif/'+str(current_batch_index)+"e_Generated_Image.png")

    #     plt.close('all')       

    # Print halve train
    # for current_batch_index in range(0,len(train_data),batch_size):
    #     test_example = train_data[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     test_example_gt  = train_gt[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     sess_results = sess.run([g_e_layer_final],feed_dict={input_binary_image:test_example})

    #     sess_results = sess_results[0][0,:,:,:]
    #     test_example = test_example[0,:,:,:]
    #     test_example_gt = test_example_gt[0,:,:,:]

    #     plt.figure()
    #     plt.imshow(np.squeeze(test_example),cmap='gray')
    #     plt.axis('off')
    #     plt.title('Original Mask ')
    #     plt.savefig('final/'+str(current_batch_index)+"a_Original_Mask.png")

    #     plt.figure()
    #     plt.imshow(np.squeeze(test_example_gt))
    #     plt.axis('off')
    #     plt.title('Ground Truth Image')
    #     plt.savefig('final/'+str(current_batch_index)+"b_Original_Image.png")

    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(np.squeeze(sess_results)   ,cmap='gray')
    #     plt.title("Generated Image")
    #     plt.savefig('final/'+str(current_batch_index)+"e_Generated_Image.png")

    #     plt.close('all')    

# -- end code --