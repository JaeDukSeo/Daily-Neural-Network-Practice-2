from autograd import grad
import numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt


def standardize(x):
    n = x.shape[0]
    mean_x = x.sum(axis=0) * (1.0/n) # D sized vector
    centered = x - mean_x
    squared = centered ** 2
    std = squared.sum(axis=0) * (1.0/n)
    istd = (std + 1e-10) ** -0.5
    x_stand = centered * istd
    return x_stand

def batchnorm_forward(x, eps=1e-10):

  N, D = x.shape

  #step1: calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #store intermediate
  cache = (xhat,xmu,ivar,sqrtvar,var,eps)

  return xhat, cache




# data
sample_data = np.array([
    [100,2],
    [20,1],
    [2,27],
    [86,0.9],
    [1,3]
])
ones_sample = np.ones_like(sample_data)

print(sample_data.mean(axis=0))
print(sample_data.std(axis=0))
# plt.scatter(sample_data[:,0],sample_data[:,0],color='green')
# plt.show()
# stand_grad = egrad(standardize)
# print(stand_grad(ones_sample))

print('----------------')
stand_data,cache = batchnorm_forward(sample_data)
print(stand_data.mean(axis=0))
print(stand_data.std(axis=0))
# plt.scatter(stand_data[:,0],stand_data[:,0],color='red')
# plt.show()

print('----------------')
my_stand = standardize(sample_data)
print(my_stand.mean(axis=0))
print(my_stand.std(axis=0))
# plt.scatter(my_stand[:,0],my_stand[:,0],color='red')
# plt.show()









# -- end code --
