import numpy as np
from scipy import linalg
import numpy as np
np.set_printoptions(precision=3,suppress=True)
# https://github.com/statsmodels/statsmodels/issues/3039
def orthogonal_complement(x, normalize=False, threshold=1e-15):
    """Compute orthogonal complement of a matrix

    this works along axis zero, i.e. rank == column rank,
    or number of rows > column rank
    otherwise orthogonal complement is empty

    TODO possibly: use normalize='top' or 'bottom'

    """
    x = np.asarray(x)
    r, c = x.shape
    if r < c:
        import warnings
        warnings.warn('fewer rows than columns', UserWarning)

    # we assume svd is ordered by decreasing singular value, o.w. need sort
    s, v, d = np.linalg.svd(x)
    rank = (v > threshold).sum()

    oc = s[:, rank:]

    if normalize:
        k_oc = oc.shape[1]
        oc = oc.dot(np.linalg.inv(oc[:k_oc, :]))
    return oc

data = np.random.randn(5,3)


U,s,V = np.linalg.svd(data,full_matrices=True)

print(U)
print(s)
print(V)

np_orth = orthogonal_complement(U)

print(U)
print(np_orth)
print(np_orth.T @ U)
print(np_orth.T @ np_orth)

print(np_orth@np_orth.T)
print(np.eye(5)  - U @ U.T)

orth_one =  np_orth@np_orth.T
orth_two =  np.eye(5)  - U @ U.T
print(np.isclose(orth_one,orth_two))


# data_you = np.array([
#     [1,-5,2,3],
#     [2,1,1,1]
# ])

# data_orth = np.array([
#     [-7,3,11,0],
#     [-8,5,0,11]
# ])
# np_orth = orthogonal_complement(data_you.T).T
# print(np_orth)
# print(data_you @ data_orth.T)
# print(data_you @ np_orth.T)

