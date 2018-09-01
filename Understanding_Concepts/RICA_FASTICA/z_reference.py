import logging
from time import time
import numpy
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

# display progress logs on stdout
logging.basicConfig(level=logging.INFO, \
    format='%(asctime)s %(levelname)s %(message)s')

# image plot setting
n_row, n_col = 2,5
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(1)

####################################
# data preparation
####################################
# Load faces data
# shuffle : is the same person
# random_state : seed for the shuffle
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

# get info of dataset
n_samples, n_features = faces.shape

# global centering
# axis = 0 (row), 1 (column)
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)

####################################
# desc : plot images
####################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

####################################
# set estimators
####################################
estimators = [
    ('Clusters - MiniBatchKMeans',
        MiniBatchKMeans(n_clusters=n_components, tol=1e-3, batch_size=20,
                        max_iter=50, random_state=rng),
     True),
    ('Independent components - FastICA',
     decomposition.FastICA(n_components=n_components, whiten=True),
     True),
    ('MiniBatchDictionaryLearning',
        decomposition.MiniBatchDictionaryLearning(n_components=15, alpha=0.1,
                                                  n_iter=50, batch_size=3,
                                                  random_state=rng),
     True)
]


####################################
# plot images
####################################
# Plot a sample of the input data
plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

# Do the estimation and plot it
for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))

    t0 = time()
    data = faces

    if center:
        data = faces_centered

    # start to fit the data
    estimator.fit(data)

    # calculate the time spent
    train_time = (time() - t0)

    print("done in %0.3fs" % train_time)

    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_

    plot_gallery('%s - Train time %.1fs' % (name, train_time), components_[:n_components])

plt.show()
