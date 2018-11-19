# make circle original data
from sklearn import  datasets
import autograd.numpy as np
from autograd import elementwise_grad as egrad

np.random.seed(6789)
n_samples   = 5
data,label  = datasets.make_blobs(n_samples=n_samples,n_features=3,cluster_std=0.3,centers=2)

U,s,V  = np.linalg.svd(data,full_matrices=False)
def get_svd(data):
    U,s,V = np.linalg.svd(data,full_matrices=False)
    S = np.diag(s)
    return U @ S @ V.T
grad_ssvd = egrad(get_svd)
print(grad_ssvd(data))

# =======================================
U,s,V  = np.linalg.svd(data,full_matrices=False)
S = np.diag(s)
fake_grad = np.ones_like(data)
gu = fake_grad @ (S @ V.T ).T; gs = np.diag(U.T @ fake_grad @ V); gv = ( U @ S).T @ fake_grad

utgu = U.T @ gu
vtgv = V   @ gv

i = np.eye(3)
f = 1/(s[...,np.newaxis,:]**2-s[...,:,np.newaxis]**2+i)-i

t1 = f * (utgu - utgu.T) * s[..., np.newaxis, :] + i * gs[..., :, np.newaxis] + s[..., :, np.newaxis] * (f * (vtgv - vtgv.T))
t1 = U @ t1 @ V

i_minus_uut = np.eye(5) - U @ U.T
t1 = t1 + i_minus_uut @ gu @ (V /s [..., :, np.newaxis])
t11 = t1.copy()
i_minus_vvt = np.eye(3) - V @ V.T

print(i_minus_vvt)
print((U / s[..., np.newaxis, :]) @ gv.T @i_minus_vvt)
t1 = t1 + (U / s[..., np.newaxis, :]) @ gv.T @i_minus_vvt

print('-------------------------')
print(
    t1 
)
print('-------------------------')
print(
    t11 
)
print('-------------------------')
print(
    np.allclose(t1,grad_ssvd(data))
)

import tensorflow as tf

def tf_svd(data):
    S,U,VT   = tf.linalg.svd(data,full_matrices=False)
    S = tf.diag(S)
    data_hat = U @ S @ tf.transpose(VT)
    return data_hat


sess = tf.InteractiveSession()
one  = tf_svd(data)
one  = one.eval() 
print(
    np.allclose(one,data)
)

# -- end code --