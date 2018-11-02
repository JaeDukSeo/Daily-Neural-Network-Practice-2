import numpy as np


s = np.array([0.004,0.0002,2])
print(sum(s))

def exp(x):
    return np.exp(x)/np.exp(x).sum()

def expt(x,t=100):
    return np.exp(x/t)/np.exp(x/t).sum()


print(np.around(exp(s),5))

print(np.around(expt(s),5))