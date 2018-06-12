import numpy as np,sys 
import matplotlib.pylab as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(678)

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
    
def load_data():
    X,Y = make_classification(n_samples = 50,n_features=3,n_redundant=0)
    return X,Y

def simple_model(X,Y):
    clf_org_x = SVC()
    clf_org_x.fit(X,Y)
    predict = clf_org_x.predict(X)
    acc=  accuracy_score(Y,predict)
    return acc

X,Y = load_data()

# X1, Y1 = make_classification(n_features=2, n_redundant=0, n_samples=300,n_informative=1,n_clusters_per_class=1)
# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, edgecolor='k')
# plt.show()

acc = simple_model(X,Y)
X_trans = sfiltering(X,2)
acc1= simple_model(X_trans,Y)
X_trans1 = sfiltering(X,1)
acc2= simple_model(X_trans1,Y)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o', c=Y)
plt.show()


plt.scatter(X_trans[:, 0], X_trans[:, 1], marker='o', c=Y, edgecolor='k')
plt.show()

plt.scatter(X_trans1[:, 0],[5] *len(X_trans1)  ,marker='o', c=Y, edgecolor='k')
plt.show()

sys.exit()

print('-----------------------')
print('Original Data Shape: ',X.shape)
print('New Data Shape: ',X_trans.shape)
print('New Data Shape: ',X_trans1.shape)

print('-----------------------')
print("Without sparsefiltering, accuracy = %f "%(acc))
print("One Layer Accuracy, = %f, Increase = %f"%(acc1,acc1-acc))
print("Two Layer Accuracy,  = %f, Increase = %f"%(acc2,acc2-acc1))


# -- end code --