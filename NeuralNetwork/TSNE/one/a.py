import numpy as np
from load_data import load_mnist
np.random.seed(1)

# Set global parameters
NUM_POINTS = 200            # Number of samples from MNIST
CLASSES_TO_USE = [0, 1, 8]  # MNIST classes to use
PERPLEXITY = 20
SEED = 1                    # Random seed
MOMENTUM = 0.9
LEARNING_RATE = 10.
NUM_ITERS = 500             # Num iterations to train for
TSNE = False                # If False, Symmetric SNE
NUM_PLOTS = 5               # Num. times to plot in training



# Load the first NUM_POINTS 0's, 1's and 8's from MNIST
X, y = load_mnist('datasets/',
                      digits_to_keep=CLASSES_TO_USE,
                      N=NUM_POINTS)

print(X.shape)
print(y.shape)

def neg_distance(X):
    X_sum = np.sum(X**2,1)
    distance = np.reshape(X_sum,[-1,1])
    return -(distance - 2*X.dot(X.T)+distance.T) 

def softmax_max(X,diag=True):
    X_exp = np.exp(X - X.max(1).reshape([-1, 1]))
    X_exp = X_exp + 1e-10
    if diag: np.fill_diagonal(X_exp, 0.)
    # X_exp = X_exp + 1e-10
    return X_exp/X_exp.sum(1).reshape([-1, 1])

def calc_prob_matrix(distances, sigmas=None):
    """Convert a distances matrix to a matrix of probabilities."""
    if sigmas is not None:
        two_sig_sq = 2. * sigmas.reshape([-1, 1]) ** 2
        return softmax(distances / two_sig_sq)
    else:
        return softmax(distances)

temp = np.array([
    [1,1],
    [2,2],
    [3,3]
])
print(neg_distance(temp))
print(softmax_max(neg_distance(temp)))



# -- end code --