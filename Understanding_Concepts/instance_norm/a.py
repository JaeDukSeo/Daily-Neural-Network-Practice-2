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

# --------- case 1 Original Batch normalize Numpy ------
def case_1_original_norm():
    
    global test_data
    case1_data = test_data[:10,:,:,:]

    mini_batch_mean = case1_data.sum(axis=0) / len(case1_data)
    mini_batch_var = ((case1_data-mini_batch_mean) ** 2).sum(axis=0) / len(case1_data)

    case_1 = (case1_data-mini_batch_mean)/ ( (mini_batch_var + 1e-8) ** 0.5 )

    plt.imshow(np.squeeze(case_1[0,:,:,:]),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(case_1[4,:,:,:]),cmap='gray')
    plt.show()

    print('\n=================================')
    print('============== Case 1 Implementation ===================')
    print("Data Shape: ",case_1.shape)
    print("Data Max: ",case_1.max())
    print("Data Min: ",case_1.min())
    print("Data Mean: ",case_1.mean())
    print("Data Variance: ",case_1.var())
    plt.hist(case_1.flatten() ,bins='auto')
    plt.show()
    print('=================================')
case_1_original_norm()
# --------- case 1 Original Batch normalize Numpy ------

# --------- case 2 Original Batch normalize TF ------
def case_2_original_norm():
    
    global test_data
    case2_data = test_data[:10,:,:,:]

    case_2 = tf.nn.batch_normalization(case2_data,mean=case2_data.mean(axis=0),variance=case2_data.var(axis=0),variance_epsilon=1e-8,offset=0.0,scale=1.0).eval()

    plt.imshow(np.squeeze(case_2[0,:,:,:]),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(case_2[4,:,:,:]),cmap='gray')
    plt.show()

    print('\n=================================')
    print('============== Case 2 Implementation ===================')
    print("Data Shape: ",case_2.shape)
    print("Data Max: ",case_2.max())
    print("Data Min: ",case_2.min())
    print("Data Mean: ",case_2.mean())
    print("Data Variance: ",case_2.var())
    plt.hist(case_2.flatten() ,bins='auto')
    plt.show()
    print('=================================')
case_2_original_norm()
# --------- case 2 Original Batch normalize TF ------

# --------- case 3 Stylization Batch normalize Numpy ------
def case_3_Stylization_norm():
    
    global test_data
    case3_data = test_data[:10,:,:,:]

    mini_batch_mean = case3_data.sum(axis=0) / (len(case3_data) * case3_data.shape[1] * case3_data.shape[2]) 
    mini_batch_var = ((case3_data-mini_batch_mean) ** 2).sum(axis=0) /(len(case3_data) * case3_data.shape[1] * case3_data.shape[2]) 

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
case_3_Stylization_norm()
# --------- case 3 Stylization Batch normalize Numpy ------

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis."""
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis."""
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


# --------- case 4 Stylization Batch normalize TF ------
# def case_4_Stylization_norm():
    
#     global test_data
#     case4_data = test_data[:10,:,:,:]

#     mu = tf.reduce_mean(case4_data, axis=[1,2], keep_dims=True).eval()
#     sigma = reduce_std(case4_data, axis=[1,2], keepdims=True).eval()

#     epsilon = 1e-8
#     case_4 = (case4_data-mu)/(sigma + epsilon)**0.5

#     plt.imshow(np.squeeze(case_4[0,:,:,:]),cmap='gray')
#     plt.show()

#     plt.imshow(np.squeeze(case_4[4,:,:,:]),cmap='gray')
#     plt.show()

#     print('\n=================================')
#     print('============== Case 3 Implementation ===================')
#     print("Data Shape: ",case_4.shape)
#     print("Data Max: ",case_4.max())
#     print("Data Min: ",case_4.min())
#     print("Data Mean: ",case_4.mean())
#     print("Data Variance: ",case_4.var())
#     plt.hist(case_4.flatten() ,bins='auto')
#     plt.show()
#     print('=================================')
# case_4_Stylization_norm()
# --------- case 4 Stylization Batch normalize TF ------

# --------- case 5 Stylization Instance normalize Numpy ------
def case_5_Stylization_norm():
    
    global test_data
    case5_data = test_data[:10,:,:,:]

    mini_batch_mean = case5_data.sum(axis=0) / ( case5_data.shape[1] * case5_data.shape[2]) 
    mini_batch_var = ((case5_data-mini_batch_mean) ** 2).sum(axis=0) /( case5_data.shape[1] * case5_data.shape[2]) 

    case_5 = (case5_data-mini_batch_mean)/ ( (mini_batch_var + 1e-8) ** 0.5 )

    plt.imshow(np.squeeze(case_5[0,:,:,:]),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(case_5[4,:,:,:]),cmap='gray')
    plt.show()

    print('\n=================================')
    print('============== Case 5 Implementation ===================')
    print("Data Shape: ",case_5.shape)
    print("Data Max: ",case_5.max())
    print("Data Min: ",case_5.min())
    print("Data Mean: ",case_5.mean())
    print("Data Variance: ",case_5.var())
    plt.hist(case_5.flatten() ,bins='auto')
    plt.show()
    print('=================================')
case_5_Stylization_norm()
# --------- case 5 Stylization Instance normalize Numpy ------

# --------- case 6 Stylization Instance normalize TF ------
def case_6_Stylization_norm():
    
    global test_data
    case4_data = test_data[:10,:,:,:]

    mu = tf.reduce_mean(case4_data, axis=[1,2], keep_dims=True).eval()
    sigma = reduce_std(case4_data, axis=[1,2], keepdims=True).eval()

    epsilon = 1e-8
    case_4 = (case4_data-mu)/(sigma + epsilon)**0.5

    plt.imshow(np.squeeze(case_4[0,:,:,:]),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(case_4[4,:,:,:]),cmap='gray')
    plt.show()

    print('\n=================================')
    print('============== Case 6 Implementation ===================')
    print("Data Shape: ",case_4.shape)
    print("Data Max: ",case_4.max())
    print("Data Min: ",case_4.min())
    print("Data Mean: ",case_4.mean())
    print("Data Variance: ",case_4.var())
    plt.hist(case_4.flatten() ,bins='auto')
    plt.show()
    print('=================================')
case_6_Stylization_norm()
# --------- case 6 Stylization Instance normalize TF ------



# ---- end code ----

















# --------- case 4 Stylization Batch normalize TF ------
def case_14_Stylization_norm():
    
    global test_data
    case4_data = test_data[:10,:,:,:]

    mean, var = tf.nn.moments(tf.convert_to_tensor(case4_data), [1, 2], keep_dims=True)
    case_4 = tf.div(tf.subtract(case4_data, mean), tf.sqrt(tf.add(var, 1e-8))).eval()

    plt.imshow(np.squeeze(case_4[0,:,:,:]),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(case_4[4,:,:,:]),cmap='gray')
    plt.show()

    print('\n=================================')
    print('============== Case 3 Implementation ===================')
    print("Data Shape: ",case_4.shape)
    print("Data Max: ",case_4.max())
    print("Data Min: ",case_4.min())
    print("Data Mean: ",case_4.mean())
    print("Data Variance: ",case_4.var())
    plt.hist(case_4.flatten() ,bins='auto')
    plt.show()
    print('=================================')
case_14_Stylization_norm()
# --------- case 4 Stylization Batch normalize TF ------