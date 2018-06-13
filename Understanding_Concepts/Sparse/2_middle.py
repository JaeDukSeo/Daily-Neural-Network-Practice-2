import numpy as np,sys 
import matplotlib.pylab as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')
# Principal component analysis
# Independent Component Analysis.
# LinearDiscriminantAnalysis
# t-distributed Stochastic Neighbor Embedding
from sklearn.decomposition import PCA,FastICA,FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

np.random.seed(678)

# Make the Data set
def load_data_clear_cut():
    X,Y = make_classification(n_samples=100,n_features=3,
                              n_redundant=0,n_clusters_per_class=1,n_classes=3,class_sep=3.2,n_informative=3)
    return X,Y

def load_data_not_so_clear():
    X,Y = make_classification(n_samples=100,n_features=3,
                              n_redundant=0,n_clusters_per_class=1,n_classes=3,class_sep=0.2,n_informative=3)
    return X,Y


# ==============================================
# Code From: https://github.com/subramgo/SparseFiltering
epsilon = 1e-8
def soft_absolute(v):
    return np.sqrt(v**2 + epsilon)

def get_objective_fn(X,n_dim,n_features):

    def _objective_fn(W):
        W = W.reshape(n_dim,n_features)
        Y = np.dot(X,W)
        Y = soft_absolute(Y)
        
        # Normalize feature across all examples
        # Divide each feature by its l2-norm
        Y = Y / np.sqrt(np.sum(Y**2,axis=0) + epsilon)        
        
        # Normalize feature per example
        Y = Y / np.sqrt(np.sum(Y**2,axis=1)[:,np.newaxis] + epsilon )
        print(np.sum(Y))
        return np.sum(Y)

    return _objective_fn

def sfiltering(X,n_features=5):
    n_samples,n_dim = X.shape
    # Intialize the weight matrix W (n_dim,n_features)
    # Intialize the bias term b(n_features)
    W = np.random.randn(n_dim,n_features)

    obj_function = get_objective_fn(X,n_dim,n_features)
    
    opt_out = minimize(obj_function,W,method='L-BFGS-B',options={'maxiter':10,'disp':False})
    W_final = opt_out['x'].reshape(n_dim,n_features)
    
    transformed_x = np.dot(X,W_final)
    return transformed_x
# ==============================================
    

# -------- clear cut difference in data ---------
# show the original data 
X,Y = load_data_clear_cut()
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o', c=Y)
# plt.title('Original Data Shape : ' + str(X.shape))
# plt.show()
# plt.close('all')

print('------- two component ------ ')
component_number = 2
X_sparse = sfiltering(X,2)
plt.scatter(X_sparse[:, 0], X_sparse[:, 1], marker='o', c=Y, edgecolor='k')
plt.title('Sprase Data Shape : ' + str(X_sparse.shape))
plt.show()

# ----------------------------------
print('------- one component ------ ')
component_number = 1
X_sparse = sfiltering(X,2)
plt.scatter(X_sparse[:, 0],[1] * len(X_sparse), marker='o', c=Y, edgecolor='k')
plt.title('Sprase Data Shape : ' + str(X_sparse.shape))
plt.show()


# -- end code --    