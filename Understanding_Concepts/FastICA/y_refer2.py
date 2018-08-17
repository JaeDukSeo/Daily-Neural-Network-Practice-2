import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# --------------------------------
cov = np.dot(X.T, X)
EPS = 10e-5
d, E = np.linalg.eigh(cov)
D = np.diag(1. / np.sqrt(d + EPS))
W = np.dot(np.dot(E, D), E.T)
X_2 = np.dot(X, W).T

def _sym_decorrelation(W):
    """ Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    one = np.sqrt(np.max(W.dot(W.T),1).sum() )
    return W / one

def _cube(x):
    return x ** 3, (3 * x ** 2).mean(axis=-1)

def _logcosh(x):
    alpha = 1.0
    x *= alpha
    gx = np.tanh(x, x)  # apply the tanh inplace
    g_x = np.empty(x.shape[0])
    # XXX compute in chunks to avoid extra allocation
    for i, gx_i in enumerate(gx):  # please don't vectorize.
        g_x[i] = (alpha * (1 - gx_i ** 2)).mean()
    return gx, g_x

def _exp(x):
    exp = np.exp(-(x ** 2) / 2)
    gx = x * exp
    g_x = (1 - x ** 2) * exp
    return gx, g_x.mean(axis=-1)

tol = 1e-4
# http://cis.legacy.ics.tkk.fi/aapo/papers/IJCNN99_tutorialweb/node30.html
W = np.random.randn(3,3).astype(np.float64)
W = _sym_decorrelation(W)
p_ = float(X_2.shape[1])
m,v = np.zeros_like(W),np.zeros_like(W)

for iter in range(50000):
    gwtx, g_wtx = _exp(W.dot(X_2))
    W1 = np.dot(gwtx, X_2.T) / p_ - g_wtx[:, np.newaxis] * W
    W1 = 3/2*W1 - 0.5 * W1.dot(W1.T).dot(W1)
    W = W1

final_A =W.dot(X_2).T

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix


# For comparison, compute PCA
# pca = PCA(n_components=3)
# H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results

plt.figure()

models = [X, S, S_, final_A]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
