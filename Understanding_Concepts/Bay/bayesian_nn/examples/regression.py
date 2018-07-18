from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import six

import numpy as np
import numpy.random as npr
import tensorflow as tf

import bayesian_nn as bnn
from bayesian_nn.layers.layers import Dense
from bayesian_nn.distributions.distributions import FactorizedGaussian


def build_toy_data(num=100):
    """Builds a toy 1-D regression with an underlying quadratic function."""

    def f(x):
        """Ground truth function."""
        return -0.9 * x**2 + 1.9 * x + 1.

    xs = [[npr.rand()] for _ in range(num)]
    ys = [[f(*x)] for x in xs]

    return np.array(xs), np.array(ys)


def main(iteration=1000, print_every=10):

    xs, ys = build_toy_data()

    # symbolic variables
    sy_x = tf.placeholder(tf.float32, shape=[None, 1])
    sy_y = tf.placeholder(tf.float32, shape=[None, 1])

    fc_1 = Dense('fc_1', 1, 100,
                 posterior=FactorizedGaussian(1, 100),
                 prior=FactorizedGaussian(1, 100, is_prior=True))
    fc_2 = Dense('fc_2', 100, 1,
                 posterior=FactorizedGaussian(100, 1),
                 prior=FactorizedGaussian(100, 1, is_prior=True))

    # two layer bayesian neural net
    h, kl_1 = fc_1(sy_x)
    p, kl_2 = fc_2(tf.nn.relu(h))

    elbo = -tf.reduce_sum((p - sy_y)**2 - kl_1 - kl_2)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(-elbo)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init_op)

        for i in range(iteration):
            sess.run(train_op, feed_dict={sy_x: xs, sy_y: ys})

            if i % print_every:
                stat = sess.run(elbo, feed_dict={sy_x: xs, sy_y: ys})
                print ('iteration [%d/%d] loss %.4f' % (i+1, iteration, stat))


if __name__ == '__main__':
    main()
