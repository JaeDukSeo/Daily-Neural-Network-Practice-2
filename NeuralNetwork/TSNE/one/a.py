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
        return softmax_max(distances / two_sig_sq)
    else:
        return softmax_max(distances)

def binary_search(eval_fn, target, tol=1e-10, max_iter=10000, 
                  lower=1e-20, upper=1000.):
    """Perform a binary search over input values to eval_fn.
    
    # Arguments
        eval_fn: Function that we are optimising over.
        target: Target value we want the function to output.
        tol: Float, once our guess is this close to target, stop.
        max_iter: Integer, maximum num. iterations to search for.
        lower: Float, lower bound of search range.
        upper: Float, upper bound of search range.
    # Returns:
        Float, best input value to function found during search.
    """
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess

def calc_perplexity(prob_matrix):
    """Calculate the perplexity of each row 
    of a matrix of probabilities."""
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix + 1e-10), 1)
    perplexity = 2.0 ** entropy
    return perplexity

def perplexity(distances, sigmas):
    """Wrapper function for quick calculation of 
    perplexity over a distance matrix."""
    return calc_perplexity(calc_prob_matrix(distances, sigmas))

def find_optimal_sigmas(distances, target_perplexity):
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role."""
    sigmas = [] 
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: perplexity(distances[i:i+1, :], np.array(sigma))
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)

print('--------')
temp = np.array([
    [1,1],
    [2,2],
    [3,3]
])
temp_distance = neg_distance(temp)
temp_sigma = find_optimal_sigmas(temp_distance,5)

print('--------')
X_distance = neg_distance(X)
X_sigma = find_optimal_sigmas(temp_distance,5)

# -- end code --