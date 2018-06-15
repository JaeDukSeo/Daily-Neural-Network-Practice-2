import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util import *

import os, sys 
np.random.seed(789)
tf.set_random_seed(7689)
plt.style.use('seaborn')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.examples.tutorials.mnist import input_data
PathDicom = "../../../Dataset/MNIST/"

mnist = input_data.read_data_sets(PathDicom, one_hot=True)

X = mnist.train._images.reshape(55000,28,28)
Y = mnist.train._labels
index = np.arange(55000)
np.random.shuffle(index)

train_bf = BatchFeeder(X[index[:54000]], Y[index[:54000]], 128)
valid_bf = BatchFeeder(X[index[54000:]], Y[index[54000:]], 32)


print(train_bf.next().shape)
print(train_bf.next().shape)
print(train_bf.next().shape)
sys.exit()

model = testnet()
model.train(train_bf, 5)

x, y = valid_bf.next()
pred = model.predict(x)
ex = model.explain(x)

index = 0

print("Truth:", np.argmax(y[index]))
print("Prediction:", np.argmax(pred[index]))

# Plot the true image.
plt.figure(figsize=(9.5,1))
plt.subplot(1,11,1)
plt.imshow(x[index].reshape(28,28), cmap="Greys")
plt.xticks([],[])
plt.yticks([],[])
plt.title("Original image", fontsize=8)

# Generate explanation with respect to each of 10 output channels.
exs = []
for i in range(10):
    exs.append(ex[i][index].reshape(28, 28))
exs = np.array(exs)

# Plot them
th = max(np.abs(np.min(exs)), np.abs(np.max(exs)))
for i in range(1,11):
    e = exs[i-1]
    plt.subplot(1,11,1+i)
    plt.imshow(e, cmap="seismic", vmin=-1*th, vmax=th)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.title("Exp. for ("+str(i-1)+")", fontsize=8)
plt.tight_layout()
plt.show()


# -- end code --