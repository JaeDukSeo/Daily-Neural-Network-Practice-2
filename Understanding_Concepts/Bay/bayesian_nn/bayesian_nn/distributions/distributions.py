from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import six

import numpy as np

import tensorflow as tf
from tensorflow.contrib.distributions import Normal


class AbstractDistribution:
    """Abstract base class for various distributions. Provides an interface."""

    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError

    def log_prob(self):
        """Called `log_prob` instead of `log_density` since we may also have
        discrete distributions."""
        raise NotImplementedError

    def __str__(self):
        return 'Abstract base class for all distributions'


class FactorizedGaussian(AbstractDistribution):
    """Factorized Gaussian distribution for a layer of weights."""

    def __init__(self, in_dims, ou_dims, **kwargs):

        self.in_dims = in_dims
        self.ou_dims = ou_dims
        self.size = [in_dims, ou_dims]
        self.is_prior = False

        if 'prior_std' in kwargs:
            self.is_prior = True
            self.prior_mean = kwargs['prior_mean'] if 'prior_mean' in kwargs else 0.
            self.prior_std = kwargs['prior_std'] if 'prior_std' in kwargs else 1.

            if not self.prior_std > 0.:
                raise ValueError('Standard deviation should be greater than 0')

        self.build_weights()

    def build_weights(self):

        if self.is_prior:
            raise Exception('Prior distribution should not be sampled from')

        self.mean = tf.Variable(tf.random_normal(
            shape=self.size, mean=0., stddev=0.1))
        self.log_std = tf.Variable(tf.random_normal(
            shape=self.size, mean=-3., stddev=0.1))

        eps = Normal(0., 1.).sample(self.size)
        self.sample = tf.multiply(tf.exp(self.log_std), eps) + self.mean

    def log_prob(self, weights):

        const = -0.5*tf.log(2.*np.pi)

        if self.is_prior:
            mean, log_std = self.prior_mean, np.log(self.log_std)
        else:
            mean = self.mean
            log_std = self.log_std

        ret = -(.5*tf.exp(-2*log_std)) * (weights-mean)**2 - log_std + const
        ret = tf.reduce_sum(ret)

        return ret


class MatrixVariateGaussian(AbstractDistribution):

    def __init__(self):
        raise NotImplementedError


class Gamma(AbstractDistribution):

    def __init__(self):
        raise NotImplementedError


class InverseGamma(AbstractDistribution):

    def __init__(self):
        raise NotImplementedError


class StudentT(AbstractDistribution):

    def __init__(self):
        raise NotImplementedError
