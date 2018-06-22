#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import linear_layer
from batch_normalize import batch_norm


class Decoder(object):
    def __init__(self, layer_list):
        self.layer_list = layer_list

        self.name_scope = u'decoder'

    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, z,  is_training, reuse = False):
        u'''
        return only logits. not sigmoid(logits).
        '''
        h = z
        with tf.variable_scope(self.name_scope, reuse = reuse):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:])):
                ret = linear_layer(h, in_dim, out_dim, i)
                h = batch_norm(ret, i, is_training)
                h = tf.nn.relu(h)

        return ret
    
if __name__ == u'__main__':
    g = Decoder([2, 100, 600, 1200, 28 * 28 * 1])
    z = tf.placeholder(tf.float32, [None, 2])
    g.set_model(z, True)
