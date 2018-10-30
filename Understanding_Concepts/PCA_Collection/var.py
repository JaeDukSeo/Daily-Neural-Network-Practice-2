import numpy as np

data = np.random.randn(3,3)
data = data - data.mean(0)
scaled_data = 3 * data
scaled_vec = np.array([[1,3,-8]])
scaled_vec_data = scaled_vec @ data

temp = scaled_vec @ np.cov(data) @ scaled_vec.T

print(np.cov(data))
print(np.cov(scaled_data)/np.cov(data))
print(np.cov(scaled_vec_data))
print(temp)



# -- end code --
