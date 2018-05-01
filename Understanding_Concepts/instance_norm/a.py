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

    mini_batch_mean = case1_data.sum(axis=0) / len(case3_data)
    mini_batch_var = ((case1_data-mini_batch_mean) ** 2).sum(axis=0) / len(case1_data)

    case_1 = (case3_data-mini_batch_mean)/ ( (mini_batch_var + 1e-8) ** 0.5 )

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
# --------- case 2 Original Batch normalize TF ------

# --------- case 3 Stylization Batch normalize Numpy ------
# --------- case 3 Stylization Batch normalize Numpy ------

# --------- case 4 Stylization Batch normalize TF ------
# --------- case 4 Stylization Batch normalize TF ------

# --------- case 5 Stylization Instance normalize Numpy ------
# --------- case 5 Stylization Instance normalize Numpy ------

# --------- case 6 Stylization Instance normalize TF ------
# --------- case 6 Stylization Instance normalize TF ------



# ---- end code ----