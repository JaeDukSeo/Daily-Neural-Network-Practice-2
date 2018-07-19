from minisom import MiniSom    
import numpy as np
import matplotlib.pyplot as plt

colors = np.array([[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])

som = MiniSom(30, 30, 3, sigma=10, learning_rate=0.2) # initialization of 6x6 SOM
# som.train_random(colors, 1000) # trains the SOM with 100 iterations
som.train_random(colors, 500) # trains the SOM with 100 iterations

print(dir(som))

print(som._weights.shape)
# Plotting the response for each pattern in the iris dataset
plt.bone()
plt.imshow(som.get_weights().astype(float))  # plotting the distance map as background
plt.show()