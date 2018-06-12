import numpy as np,sys 
import matplotlib.pylab as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Principal component analysis
# Independent Component Analysis.
# LinearDiscriminantAnalysis
# t-distributed Stochastic Neighbor Embedding
from sklearn.decomposition import PCA,FastICA,FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
np.random.seed(678)

def load_data_clear_cut():
    X,Y = make_classification(n_samples=100,n_features=3,
                              n_redundant=0,n_clusters_per_class=1,n_classes=3,class_sep=3.2,n_informative=3)
    return X,Y

def load_data_not_so_clear():
    X,Y = make_classification(n_samples=100,n_features=3,
                              n_redundant=0,n_clusters_per_class=1,n_classes=3,class_sep=0.2,n_informative=3)
    return X,Y


# show the original data 
X,Y = load_data_clear_cut()
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o', c=Y)
# plt.show()

print('------- two component ------ ')
component_number = 2
pca = PCA(n_components=component_number)
X_pca = pca.fit_transform(X)
print(X_pca.shape)

ica = FastICA(n_components=component_number)
X_ica = ica.fit_transform(X)
print(X_ica.shape)

lda = LinearDiscriminantAnalysis(n_components=component_number)
X_lda = lda.fit_transform(X,Y)
print(X_lda.shape)

tsne = TSNE(n_components=component_number)
X_tsne = tsne.fit_transform(X,Y)
print(X_tsne.shape)

print('------- one component ------ ')
component_number = 1
pca = PCA(n_components=component_number)
X_pca = pca.fit_transform(X)
print(X_pca.shape)

ica = FastICA(n_components=component_number)
X_ica = ica.fit_transform(X)
print(X_ica.shape)

lda = LinearDiscriminantAnalysis(n_components=component_number)
X_lda = lda.fit_transform(X,Y)
print(X_lda.shape)

tsne = TSNE(n_components=component_number)
X_tsne = tsne.fit_transform(X,Y)
print(X_tsne.shape)

# plt.scatter(X_trans[:, 0], X_trans[:, 1], marker='o', c=Y, edgecolor='k')
# plt.show()

# plt.scatter(X_trans1[:, 0],[5] *len(X_trans1)  ,marker='o', c=Y, edgecolor='k')
# plt.show()

# print('-----------------------')
# print('Original Data Shape: ',X.shape)
# print('New Data Shape: ',X_trans.shape)
# print('New Data Shape: ',X_trans1.shape)



# -- end code --