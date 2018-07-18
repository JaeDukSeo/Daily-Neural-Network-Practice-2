# bayesian_nn
bayesian_nn is a lightweight [*Bayesian neural network*]() library built on top of tensorflow to ease network training via 
[*variational inference* (VI)](https://en.wikipedia.org/wiki/Variational_Bayesian_methods). The library is intended to resemble [slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) and help avoid massive boilerplate code. The end goal is to facilitate speedy development of Bayesian neural net models in the case where multiple stacked layers are required.

**Note: This project is currently under a major re-write!**

## Installation
```bash
pip install bayesian_nn
```

## Usage
```python
import bayesian_nn as bnn
```

## How are Bayesian neural nets trained with VI?
![](assets/bbb_demo.gif)

Bayesian neural networks are just like ordinary neural networks except that weights are given an explicit prior distribution, and the inferred posterior distribution given the training data is used to make predictions on new data. In addition to help avoid overfitting, Bayesian neural nets also give predictive uncertainty.

When making predictions, the model takes in account all weight configurations according to its posterior distribution. In practice the expectation could be approximated through Monte Carlo.

In addition, the posterior distribution of the weights could be approximated through variational inference, where the evidence/variational lower bound (or negative variational free energy) is optimized so that the KL-divergence from the approximate to true posterior is minimized.

## Layers
bayesian_nn primarily provides the user with the flexibility of stacking neural net layers where weights follow an approximate posterior distribution. But layers could degrade to MAP or ordinary point-estimation if no distribution is specified.
<!-- 
Pre-implemented layers include:

Layer | bayesian_nn
------- | --------
FullyConnected | [bnn.fully_connected]()
Conv2d | [bnn.conv2d]()
Conv2dTranspose (Deconv) | [bnn.conv2d_transpose]()
RNN | [bnn.rnn]() -->

Below is a toy example of a 2-layer (excluding input layer) Bayesian neural net for 1D regression:

```python
x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

p, q = FactorizedGaussian(isPrior=True), FactorizedGaussian([1, 100])
fc_1 = Dense('fc_1', 1, 100, prior=p, posterior=q)

p, q = FactorizedGaussian(isPrior=True), FactorizedGaussian([100, 1])
fc_2 = Dense('fc_2', 100, 1, prior=p, posterior=q)

h, kl_1 = fc_1(x)
p, kl_2 = fc_2(tf.nn.relu(h))

elbo = - tf.reduce_sum((y - p) ** 2)  - kl_1 - kl_2               # evidence lower bound

train_op = tf.train.AdamOptimizer(learning_rate).minimize(-elbo)
...
```

In the above example, kl_1 and kl_2 are the KL divergences between the approximate posterior distribution and 
prior distribution for the input-to-hidden and hidden-to-output weights respectively. 
The special case of variational learning in Bayesian neural nets with both factorized 
Gaussian prior and posterior is also referred to as 
[Bayes by Backprop](https://arxiv.org/abs/1505.05424).

Additionally, there is the flexibility to use arbitrary prior and approximate 
posterior distributions, so long as it is possible to evaluate the density (or probability mass 
with discrete models) of the prior 
as well as the approximate posterior at samples from the latter. 
To achieve this we only need specify different distributions, e.g.:

```python
...
p, q = GroupHorseShoe(isPrior=True), FactorizedGaussian([1, 100])
fc_1 = Dense('fc_1', 1, 100, prior=p, posterior=q)

p, q = FactorizedGaussian(isPrior=True), FactorizedGaussian([100, 1])
fc_2 = Dense('fc_2', 100, 1, prior=p, posterior=q)

h, kl_1 = fc_1(x)
p, kl_2 = fc_2(tf.nn.relu(h))

elbo = - tf.reduce_sum((y - p) ** 2)  - kl_1 - kl_2               # evidence lower bound

train_op = tf.train.AdamOptimizer(learning_rate).minimize(-elbo)
...
```
