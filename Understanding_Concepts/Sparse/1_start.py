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
from sklearn.decomposition import PCA,FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

np.random.seed(678)

# Make Data set
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
plt.show()
plt.close('all')


print('------- two component ------ ')
component_number = 2
pca = PCA(n_components=component_number)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=Y, edgecolor='k')
plt.title('PCA Data Shape : ' + str(X_pca.shape))
plt.show()

ica = FastICA(n_components=component_number)
X_ica = ica.fit_transform(X)
plt.scatter(X_ica[:, 0], X_ica[:, 1], marker='o', c=Y, edgecolor='k')
plt.title('FastICA Data Shape : ' + str(X_pca.shape))
plt.show()

lda = LinearDiscriminantAnalysis(n_components=component_number)
X_lda = lda.fit_transform(X,Y)
plt.scatter(X_lda[:, 0], X_lda[:, 1], marker='o', c=Y, edgecolor='k')
plt.title('LinearDiscriminantAnalysis Data Shape : ' + str(X_pca.shape))
plt.show()

tsne = TSNE(n_components=component_number)
X_tsne = tsne.fit_transform(X,Y)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='o', c=Y, edgecolor='k')
plt.title('TSNE Data Shape : ' + str(X_pca.shape))
plt.show()


# ----------------------------------
print('------- one component ------ ')
component_number = 1
pca = PCA(n_components=component_number)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=Y, edgecolor='k')
plt.title('PCA Data Shape : ' + str(X_pca.shape))
plt.show()

ica = FastICA(n_components=component_number)
X_ica = ica.fit_transform(X)
plt.scatter(X_ica[:, 0], X_ica[:, 1], marker='o', c=Y, edgecolor='k')
plt.title('FastICA Data Shape : ' + str(X_pca.shape))
plt.show()

lda = LinearDiscriminantAnalysis(n_components=component_number)
X_lda = lda.fit_transform(X,Y)
plt.scatter(X_lda[:, 0], X_lda[:, 1], marker='o', c=Y, edgecolor='k')
plt.title('LinearDiscriminantAnalysis Data Shape : ' + str(X_pca.shape))
plt.show()

tsne = TSNE(n_components=component_number)
X_tsne = tsne.fit_transform(X,Y)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='o', c=Y, edgecolor='k')
plt.title('TSNE Data Shape : ' + str(X_pca.shape))
plt.show()



# -- end code --