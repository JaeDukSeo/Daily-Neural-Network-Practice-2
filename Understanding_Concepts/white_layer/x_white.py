import numpy as np
import matplotlib.pyplot as plt
class whiten_layer():

    def feedforward(self,input):
        """
        Applies ZCA whitening to the data (X)
        http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

        X: numpy 2d array
            input data, rows are data points, columns are features

        Returns: ZCA whitened 2d array
        """
        EPS = 10e-5
        #   covariance matrix
        cov = np.dot(X.T, X)
        #   d = (lambda1, lambda2, ..., lambdaN)
        d, E = np.linalg.eigh(cov)
        #   D = diag(d) ^ (-1/2)
        D = np.diag(1. / np.sqrt(d + EPS))
        #   W_zca = E * D * E.T
        self.W = np.dot(np.dot(E, D), E.T)

        X_white = np.dot(X, self.W)

        return X_white

    def backprop(self,grad):
        return grad.dot(self.W.T)


C = np.array([[12., 2.],
              [2., 1.]])

X = np.random.multivariate_normal([0, 0], C, 2000).T
p1 = (0, -2)
p2 = (5, 1)

def plot_points(X, p1, p2):
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(10, 10))
    plt.plot(X[0, :], X[1, :], 'k.', ms=1)
    plt.hold(1)
    plt.grid(1)
    for (p, color) in [(p1, 'green'), (p2, 'red')]:
        plt.plot(p[0], p[1], color[0] + '.', ms=6)
        aprops = {'width': 0.5, 'headwidth': 5, 'color': color,
                  'shrink': 0.0}
        plt.annotate('', p, (0, 0), arrowprops=aprops)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()

plot_points(X, p1, p2)
wlayer = whiten_layer()
X_temp = wlayer.feedforward(X)
plot_points(X_temp, p1, p2)



# -- end code --
