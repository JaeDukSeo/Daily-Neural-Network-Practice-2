from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import six

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavi


class Dense:
    """Dense layer without activation applied."""

    def __init__(self, name, in_dims, ou_dims, **kwargs):
        """Initialize the layer with weights and biases.
        Args:
            in_dims: input dimensionality
            ou_dims: output dimensionality
            kwargs: Contains information about the prior and
                    approximate posterior distribution
                    e.g. could be prior and posterior, or inference_pkg
                    Note: if only prior is specified, the model performs MAP
                          if neither prior or posterior is specified, the model
                          does ordinary non-Bayesian point-estimation

        TODO: add distribution for biases
        """

        self.in_dims = in_dims
        self.ou_dims = ou_dims

        with tf.variable_scope(name):
            # the biases are trained with point estimation
            self.biases = tf.get_variable(
                'biases', [self.ou_dims], initializer=xavi())

        if 'prior' in kwargs and 'posterior' in kwargs:
            self.prior = kwargs['prior']
            self.posterior = kwargs['posterior']
            self.inference_type = 'bayesian'  # bayesian approx. inference
            self.weights = self.posterior.sample

        elif 'prior' in kwargs:
            self.prior = prior
            self.inference_type = 'map'  # maximum a posteriori

        elif 'inference_pkg' in kwargs:
            if 'posterior' in kwargs:
                raise KeyError(
                    'Should not specify both posterior and inference_pkg')
            self.inference_pkg = kwargs['inference_pkg']
            self.inference_type = 'pkg'
            self.weights = inference_pkg.posterior.sample

        else:
            self.inference_type = 'pe'  # point-estimation

        if self.inference_type == 'pe' or self.inference_type == 'map':
            with tf.variable_scope(name):
                self.weights = tf.get_variable(
                    'weights', [self.in_dims, self.ou_dims], initializer=xavi())

    def __call__(self, x):
        """Apply the dense layer to inputs.
        e.g.:
        x = tf.placeholder([None, 100], dtype=tf.float32)
        layer = Dense(100, 100, prior=Gaussian(), posterior=Gaussian())
        y = layer(x)

        Args:
            x: tensor with dim [None, self.in_dims]
        Returns:
            y: tensor with dim [None, self.ou_dims]
            kl: KL divergence between the approx. posterior and prior
        """

        y = tf.matmul(x, self.weights) + self.biases

        if self.inference_type == 'bayesian':
            logp = self.prior.log_prob(self.weights)
            logq = self.posterior.log_prob(self.weights)

        elif self.inference_type == 'pkg':
            logp = self.inference_pkg.logp
            logq = self.inference_pkg.logq

        elif self.inference_type == 'map':
            logp = prior.log_prob(self.weights)
            logq = 0.
        else:
            logp = logq = 0.
        
        kl = logp - logq

        return y, kl

    def __str__(self):
        """Return a string representation, specifying the inference type."""

        description = {
            'bayesian': 'Approximate Bayesian inference',
            'map': 'Maximum a posteriori',
            'pkg': 'Approximate Bayesian inference',
            'pe': 'Point estimation'
        }

        return 'Dense layer with {} input units and {} output units \
                    trained with {}'.format(self.in_dims,
                                            self.ou_dims,
                                            description[self.inference_type])


class Convolution:

    def __init__(self):
        raise NotImplementedError


class Recurrent:

    def __init__(self):
        raise NotImplementedError
