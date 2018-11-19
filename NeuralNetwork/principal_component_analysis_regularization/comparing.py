# make circle original data
from sklearn import  datasets
import autograd.numpy as np
from autograd import elementwise_grad as egrad

np.random.seed(6789)
n_samples   = 4
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
gu = fake_grad @ (S @ V.T ).T
gs = np.diag(U.T @ fake_grad @ V)
gv = ( U @ S).T @ fake_grad

utgu = U.T @ gu
vtgv = V @ gv

i_minus_uut = np.eye(4) - U @ U.T

i = np.eye(3)
f = 1/(s[...,np.newaxis,:]**2-s[...,:,np.newaxis]**2+i)

t1 = f * (utgu - utgu.T) * s[..., np.newaxis, :]
t1 = t1 + i * gs[..., :, np.newaxis]
t1 = t1 + s[..., :, np.newaxis] * (f * (vtgv - vtgv.T))
t1 = U @ t1 @ V
t1 = t1 + i_minus_uut @ gu @ (V / s[..., :, np.newaxis])

print('-------------------------')
print(f * (utgu - utgu.T) * s[..., np.newaxis, :])
print(
    f * (utgu * s[..., np.newaxis, :] - utgu.T* s[..., np.newaxis, :])
)

print('-------------------------')


print(
    t1
)



# -- end code --