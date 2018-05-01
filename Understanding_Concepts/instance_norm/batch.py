import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(678)
tf.set_random_seed(678)
config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.InteractiveSession(config=config)

# create data
global  test_data
test_data = np.zeros((30,32,32,1))
for i in range(30):
    new_random_image = np.random.randn(32,32) * np.random.randint(5) + np.random.randint(60)
    new_random_image = np.expand_dims(new_random_image,axis=2)
    test_data[i,:,:,:] = new_random_image

# Show Sample Data here and there
plt.imshow(np.squeeze(test_data[0,:,:,:]),cmap='gray')
plt.show()

plt.imshow(np.squeeze(test_data[4,:,:,:]),cmap='gray')
plt.show()

# 0.Print the information about the given batch of image
print('\n=================================')
print("Data Shape: ",test_data.shape)
print("Data Max: ",test_data.max())
print("Data Min: ",test_data.min())
print("Data Mean: ",test_data.mean())
print("Data Variance: ",test_data.var())
plt.hist(test_data.flatten() ,bins='auto')
plt.show()
print('=================================')

# --------- case 1 normalize whole data ------
def case_1_normalize():
    
    global test_data

    step_1 = (test_data - test_data.min(axis=0)) / (test_data.max(axis=0) - test_data.min(axis=0))
    plt.imshow(np.squeeze(step_1[0,:,:,:]),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(step_1[4,:,:,:]),cmap='gray')
    plt.show()

    print('\n=================================')
    print('============== Case 1 ===================')
    print("Data Shape: ",step_1.shape)
    print("Data Max: ",step_1.max())
    print("Data Min: ",step_1.min())
    print("Data Mean: ",step_1.mean())
    print("Data Variance: ",step_1.var())
    plt.hist(step_1.flatten() ,bins='auto')
    plt.show()
    print('=================================')
case_1_normalize()
# --------- step 1 normalize whole data ------

# --------- case 2 normalize whole data ------
def case_2_Standardization():
    
    global test_data

    case_2 = (test_data - test_data.mean(axis=0)) / test_data.std(axis=0)
    plt.imshow(np.squeeze(case_2[0,:,:,:]),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(case_2[4,:,:,:]),cmap='gray')
    plt.show()

    print('\n=================================')
    print('============== Case  ===================')
    print("Data Shape: ",case_2.shape)
    print("Data Max: ",case_2.max())
    print("Data Min: ",case_2.min())
    print("Data Mean: ",case_2.mean())
    print("Data Variance: ",case_2.var())
    plt.hist(case_2.flatten() ,bins='auto')
    plt.show()
    print('=================================')
case_2_Standardization()
# --------- case 2 normalize whole data ------

# --------- case 3 batch normalize first 10------
def case_3_batch_norm_implement():
    
    global test_data
    case3_data = test_data[:10,:,:,:]

    mini_batch_mean = case3_data.sum(axis=0) / len(case3_data)
    mini_batch_var = ((case3_data-mini_batch_mean) ** 2).sum(axis=0) / len(case3_data)
    case_3 = (case3_data-mini_batch_mean)/ ( (mini_batch_var + 1e-8) ** 0.5 )

    plt.imshow(np.squeeze(case_3[0,:,:,:]),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(case_3[4,:,:,:]),cmap='gray')
    plt.show()

    print('\n=================================')
    print('============== Case 3 Implementation ===================')
    print("Data Shape: ",case_3.shape)
    print("Data Max: ",case_3.max())
    print("Data Min: ",case_3.min())
    print("Data Mean: ",case_3.mean())
    print("Data Variance: ",case_3.var())
    plt.hist(case_3.flatten() ,bins='auto')
    plt.show()
    print('=================================')
case_3_batch_norm_implement()
# --------- case 3 batch normalize first 10------
def case_3_batch_norm_tensorflow():
    
    global test_data
    case3_data = test_data[:10,:,:,:]

    case_3 = tf.nn.batch_normalization(case3_data,
                    mean = case3_data.mean(axis=0),
                    variance = case3_data.var(axis=0),
                    offset = 0.0,scale = 1.0,
                    variance_epsilon = 1e-8
    ).eval()

    plt.imshow(np.squeeze(case_3[0,:,:,:]),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(case_3[4,:,:,:]),cmap='gray')
    plt.show()

    print('\n=================================')
    print('============== Case 3 Tensorflow ===================')
    print("Data Shape: ",case_3.shape)
    print("Data Max: ",case_3.max())
    print("Data Min: ",case_3.min())
    print("Data Mean: ",case_3.mean())
    print("Data Variance: ",case_3.var())
    plt.hist(case_3.flatten() ,bins='auto')
    plt.show()
    print('=================================')
case_3_batch_norm_tensorflow()
# --------- case 3 batch normalize first 10------