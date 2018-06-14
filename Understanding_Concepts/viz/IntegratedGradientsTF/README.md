# IntegratedGradientsTF

Tenforflow implementation of integrated gradients [1]. The algorithm "explains" a prediction of a Keras-based deep learning model by approximating Aumann–Shapley values and for the input features. These values allocate the difference between the model prediction for a reference value (all zeros by default) and the prediction for the current sample among the input features. 

# Usage

Lets say you have the following tensorflow code for basic MLP:
``` Python
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
with tf.variable_scope("predictor"):
    dense = x
    for dim in self.dimensions:
        dense = tf.contrib.slim.fully_connected(dense, dim, activation_fn=tf.nn.relu)
    dense = tf.contrib.slim.fully_connected(dense, 10, activation_fn=tf.identity)
    prediction = tf.nn.softmax(dense)
```

Add the following code to build integrated gradients into your model:
```Python
import integrated_gradients_tf as ig

inter, stepsize, ref = ig.linear_inpterpolation(x, num_steps=50)
with tf.variable_scope("predictor", reuse=True):
    dense2 = inter
    for dim in self.dimensions:
        dense2 = tf.contrib.slim.fully_connected(dense2, dim, activation_fn=tf.nn.relu)
    dense2 = tf.contrib.slim.fully_connected(dense2, 10, activation_fn=tf.identity)
    prediction2 = tf.nn.softmax(dense2)
    
explanations = []
for i in range(10):
    explanations.append(ig.build_ig(inter, stepsize, prediction2[:, i], num_steps=50))
```

# Tips:
- Make sure to set ```reuse=True``` to reuse weights
- See example.ipynb and util.py for more complete example.
- See https://github.com/hiranumn/IntegratedGradients for Keras implementation of this.
- You can easily change your code to explain hidden layer activations.

# References
1. Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic Attribution for Deep Networks." arXiv preprint arXiv:1703.01365 (2017).

Email me at hiranumn at cs dot washington dot edu for questions.
