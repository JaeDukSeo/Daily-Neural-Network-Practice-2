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
from skimage.color import rgba2rgb

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(678)
tf.set_random_seed(678)
ia.seed(678)

def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf.cast(tf.greater(x,0),tf.float32)  + (tf_elu(tf.cast(tf.less_equal(x,0),tf.float32) * x) + 1.0)
def tf_softmax(x): return tf.nn.softmax(x)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# code from: https://github.com/tensorflow/tensorflow/issues/8246
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

# ================= VIZ =================
def show_hist_of_weigt(all_weight_list,status='before'):
    fig = plt.figure()
    weight_index = 0

    for i in range(1,1+int(len(all_weight_list)//3)):
        ax = fig.add_subplot(1,4,i)
        ax.grid(False)
        temp_weight_list = all_weight_list[weight_index:weight_index+3]
        for temp_index in range(len(temp_weight_list)):
            current_flat = temp_weight_list[temp_index].flatten()
            ax.hist(current_flat,histtype='step',bins='auto',label=str(temp_index+weight_index))
            ax.legend()
        ax.set_title('From Layer : '+str(weight_index+1)+' to '+str(weight_index+3))
        weight_index = weight_index + 3
    plt.savefig('viz/weights_'+str(status)+"_training.png")
    plt.close('all')

# Def: Simple function to show 9 image with different channels
def show_9_images(image,layer_num,image_num,channel_increase=3,alpha=None,gt=None,predict=None):
    image = (image-image.min())/(image.max()-image.min())
    fig = plt.figure()
    color_channel = 0
    limit = 10
    if alpha: limit = len(gt)
    for i in range(1,limit):
        ax = fig.add_subplot(3,3,i)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if alpha:
            ax.set_title("GT: "+str(gt[i-1])+" Predict: "+str(predict[i-1]))
        else:
            ax.set_title("Channel : " + str(color_channel) + " : " + str(color_channel+channel_increase-1))
        ax.imshow(np.squeeze(image[:,:,color_channel:color_channel+channel_increase]))
        color_channel = color_channel + channel_increase
    
    if alpha:
        plt.savefig('viz/z_'+str(alpha) + "_alpha_image.png")
    else:
        plt.savefig('viz/'+str(layer_num) + "_layer_"+str(image_num)+"_image.png")
    plt.close('all')
# ================= VIZ =================

# ================= DATA AUGMENTATION =================
seq = iaa.Sequential([
    iaa.Sometimes(0.5,
        iaa.Fliplr(0.5), # horizontal flips
    ),
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Sometimes(0.5,
    iaa.Affine(
        rotate=(-180, 180),
    ))
], random_order=True) # apply augmenters in random order
# ================= DATA AUGMENTATION =================

# ================= LAYER CLASSES =================
class CNN():
    
    def __init__(self,k,inc,out,stddev):
        self.w = tf.Variable(tf.truncated_normal([k,k,inc,out],stddev=stddev))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding) 
        self.layerA = tf_elu(self.layer)
        return self.layerA 

    def backprop(self,gradient,learning_rate_change,stride=1,padding='SAME'):
        grad_part_1 = gradient 
        grad_part_2 = d_tf_elu(self.layer) 
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.conv2d_backprop_filter(input = grad_part_3,filter_sizes = self.w.shape,out_backprop = grad_middle,
            strides=[1,stride,stride,1],padding=padding
        )

        grad_pass = tf.nn.conv2d_backprop_input(input_sizes = [batch_size] + list(grad_part_3.shape[1:]),filter= self.w,out_backprop = grad_middle,
            strides=[1,stride,stride,1],padding=padding
        )

        update_w = []
        update_w.append(tf.assign( self.m,self.m*beta1 + (1-beta1) * (grad)   ))
        update_w.append(tf.assign( self.v_prev,self.v_prev*beta2 + (1-beta2) * (grad ** 2)   ))
        m_hat = self.m / (1-beta1)
        v_hat = self.v_prev / (1-beta2)
        adam_middel = learning_rate_change/(tf.sqrt(v_hat) + adam_e)
        update_w.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat)  )))         

        return grad_pass,update_w 

class CNN_Trans():
    
    def __init__(self,k,inc,out,stddev):
        self.w = tf.Variable(tf.truncated_normal([k,k,inc,out],stddev=stddev))
        self.m,self.v_prev = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return self.w

    def feedforward(self,input,stride=1,padding='SAME'):
        self.input  = input
        output_shape = self.input.shape[2].value * 2
        self.layer  = tf.nn.conv2d_transpose(
            input,self.w,output_shape=[batch_size,output_shape,output_shape,self.w.shape[3].value],
            strides=[1,stride,stride,1],padding=padding) 
        self.layerA = tf_elu(self.layer)
        return self.layerA 
# ================= LAYER CLASSES =================

# stl 10 data
PathDicom = "../../Dataset/STL10/stl10_binary/"

train_file     = open(PathDicom+"train_X.bin",'rb')
train_label_f  = open(PathDicom+"train_y.bin",'rb')

test_file     = open(PathDicom+"test_X.bin",'rb')
test_label_f  = open(PathDicom+"test_y.bin",'rb')

# data emoji
PathDicom = "../../Dataset/emoji/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if not ".webloc" in filename.lower() and not  ".meta" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

emoji_images = np.zeros((len(lstFilesDCM),40,40,4))
for x in range(len(lstFilesDCM)):
    temp = imresize(imread(lstFilesDCM[x]),(40,40,4))
    emoji_images[x,:,:,:] = temp

# one hot encode and transpose
onehot_encoder = OneHotEncoder(sparse=True)
x_data = np.fromfile(train_file, dtype=np.uint8)
x_data = np.reshape(x_data, (-1, 3, 96, 96))
x_data = np.transpose(x_data, (0, 3, 2, 1))
train_label_pure = np.expand_dims(np.fromfile(train_label_f, dtype=np.uint8).astype(np.float64),axis=1)
train_label = onehot_encoder.fit_transform(train_label_pure).toarray().astype(np.float32)

y_data = np.fromfile(test_file, dtype=np.uint8)
y_data = np.reshape(y_data, (-1, 3, 96, 96))
y_data = np.transpose(y_data, (0, 3, 2, 1))
test_label = np.expand_dims(np.fromfile(test_label_f, dtype=np.uint8).astype(np.float64),axis=1)
test_label = onehot_encoder.fit_transform(test_label).toarray().astype(np.float32)

train_batch = np.zeros((len(x_data),96,96,3))
test_batch = np.zeros((len(y_data),96,96,3))

for x in range(len(x_data)):
    train_batch[x,:,:,:] = imresize(x_data[x,:,:,:],(96,96))
for x in range(len(y_data)):
    test_batch[x,:,:,:] =  imresize(y_data[x,:,:,:],(96,96))

train_batch = train_batch[:100]
# print out the data shape
print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)
train_label = train_batch.copy()

# code: https://pytech-solution.blogspot.com/2017/07/alphablending.html
# function to overlay a transparent image on background.
def transparentOverlay(src , overlay , pos=(0,0),scale = 1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src

for current_image in range(len(train_batch)):
    random_emoji = np.random.randint(0,len(emoji_images))
    temp = seq.augment_image(emoji_images[random_emoji])
    range_random1 = np.random.randint(0,train_batch[current_image].shape[0] - temp.shape[0] )
    range_random2 = np.random.randint(0,train_batch[current_image].shape[0] - temp.shape[0] )
    temp2 = transparentOverlay(train_batch[current_image],temp,(range_random1,range_random2))
    train_batch[current_image] = temp2

# simple normalize
train_batch = train_batch/255.0
train_label = train_label/255.0
test_batch = test_batch/255.0

# hyper parameter
num_epoch = 51
batch_size = 10
print_size = 1

learning_rate = 0.00001
learnind_rate_decay = 0.0
beta1,beta2,adam_e = 0.9,0.99,1e-8

# define class here
l1 = CNN(3,3,100,stddev=0.04)
l2 = CNN(3,100,150,stddev=0.05)
l3 = CNN(3,150,200,stddev=0.06)
l4 = CNN(3,200,250,stddev=0.04)

l5 = CNN_Trans(3,250,200,stddev=0.05)
l6 = CNN_Trans(3,200,150,stddev=0.06)
l7 = CNN_Trans(3,150,100,stddev=0.06)
l8 = CNN_Trans(3,100,3,stddev=0.05)

# graph
x = tf.placeholder(shape=[batch_size,96,96,3],dtype=tf.float32)
y = tf.placeholder(shape=[batch_size,96,96,3],dtype=tf.float32)

iter_variable = tf.placeholder(tf.float32, shape=())
learning_rate_dynamic  = tf.placeholder(tf.float32, shape=())
learning_rate_change = learning_rate_dynamic * (1.0/(1.0+learnind_rate_decay*iter_variable))
phase = tf.placeholder(tf.bool)

layer1 = l1.feedforward(x)
layer2_Input = tf.nn.avg_pool(layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2 = l2.feedforward(layer2_Input) 
layer3_Input = tf.nn.avg_pool(layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3 = l3.feedforward(layer3_Input)
layer4_Input = tf.nn.avg_pool(layer3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4 = l4.feedforward(layer4_Input)

layer5_Input = tf.nn.avg_pool(layer4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer5 = l5.feedforward(layer5_Input)
layer6 = l6.feedforward(layer5)
layer7 = l7.feedforward(layer6)
layer8 = l8.feedforward(layer7)

cost = tf.reduce_mean(tf.square(layer8-y))
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# sess
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    train_cota,train_acca = 0,0
    train_cot,train_acc = [],[]
    
    test_cota,test_acca = 0,0
    test_cot,test_acc = [],[]

    # start the training
    for iter in range(num_epoch):

        train_batch,train_label = shuffle(train_batch,train_label)
        which_grad,which_grad_print = None,None

        for batch_size_index in range(0,len(train_batch),batch_size//2):
            current_batch = train_batch[batch_size_index:batch_size_index+batch_size//2]
            current_batch_label = train_label[batch_size_index:batch_size_index+batch_size//2]

            # online data augmentation here and standard normalization
            images_aug  = seq.augment_images(current_batch.astype(np.float32))
            current_batch = np.vstack((current_batch,images_aug)).astype(np.float32)
            current_batch_label = np.vstack((current_batch_label,current_batch_label)).astype(np.float32)
            current_batch,current_batch_label  = shuffle(current_batch,current_batch_label)
            # online data augmentation here and standard normalization

            # Select which back prop we should use
            which_grad = grad_update_0
            which_grad_print = 0

            sess_result = sess.run([cost,accuracy,correct_prediction,which_grad],
            feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,learning_rate_dynamic:learning_rate,phase:True})
            print("Current Iter : ",iter," Using grad: ",which_grad_print ," current batch: ",batch_size_index, ' Current cost: ', sess_result[0],' Current Acc: ', sess_result[1],end='\r')
            train_cota = train_cota + sess_result[0]
            train_acca = train_acca + sess_result[1]

        for test_batch_index in range(0,len(test_batch),batch_size):
            current_batch = test_batch[test_batch_index:test_batch_index+batch_size].astype(np.float32)
            current_batch_label = test_label[test_batch_index:test_batch_index+batch_size].astype(np.float32)
            sess_result = sess.run([cost,accuracy,correct_prediction],
            feed_dict={x:current_batch,y:current_batch_label,iter_variable:iter,phase:False})
            print("Current Iter : ",iter, " current batch: ",test_batch_index, ' Current cost: ', sess_result[0],
            ' Current Acc: ', sess_result[1],end='\r')
            test_acca = sess_result[1] + test_acca
            test_cota = sess_result[0] + test_cota

        if iter % print_size==0:
            print("\n---------- Learning Rate : ", learning_rate * (1.0/(1.0+learnind_rate_decay*iter)) )
            print(" Using grad: ",which_grad_print,'Train Current cost: ', train_cota/(len(train_batch)/(batch_size//2)),' Current Acc: ', train_acca/(len(train_batch)/(batch_size//2) ),end='\n')
            print('Test Current cost: ', test_cota/(len(test_batch)/batch_size),' Current Acc: ', test_acca/(len(test_batch)/batch_size),end='\n')
            print("----------")

        train_acc.append(train_acca/(len(train_batch)/(batch_size//2)))
        train_cot.append(train_cota/(len(train_batch)/(batch_size//2)))
        test_acc.append(test_acca/(len(test_batch)/batch_size))
        test_cot.append(test_cota/(len(test_batch)/batch_size))
        test_cota,test_acca = 0,0
        train_cota,train_acca = 0,0

    # Normalize the cost of the training
    train_cot = (train_cot-min(train_cot) ) / (max(train_cot)-min(train_cot))
    test_cot = (test_cot-min(test_cot) ) / (max(test_cot)-min(test_cot))

    # plot the training and testing graph
    plt.figure()
    plt.plot(range(len(train_acc)),train_acc,color='red',label='acc ovt')
    plt.plot(range(len(train_cot)),train_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Train Average Accuracy / Cost Over Time")
    plt.savefig("viz/Case Train.png")

    plt.figure()
    plt.plot(range(len(test_acc)),test_acc,color='red',label='acc ovt')
    plt.plot(range(len(test_cot)),test_cot,color='green',label='cost ovt')
    plt.legend()
    plt.title("Test Average Accuracy / Cost Over Time")
    plt.savefig("viz/Case Test.png")
    plt.close('all')

# -- end code --