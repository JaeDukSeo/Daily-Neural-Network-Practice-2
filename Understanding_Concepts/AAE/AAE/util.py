#! -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
# import cv2
import numpy as np

def get_weights(name, shape, stddev, trainable = True):
    return tf.get_variable('weights{}'.format(name), shape,
                           initializer = tf.random_normal_initializer(stddev = stddev),
                           trainable = trainable)

def get_biases(name, shape, value, trainable = True):
    return tf.get_variable('biases{}'.format(name), shape,
                           initializer = tf.constant_initializer(value),
                           trainable = trainable)

def get_dim(target):
    dim = 1
    for d in target.get_shape()[1:].as_list():
        dim *= d
    return dim


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear_layer(x, in_dim, out_dim, l_id):
    weights = get_weights(l_id, [in_dim, out_dim], 1.0/np.sqrt(float(in_dim)))
    biases  = get_biases(l_id, [out_dim], 0.0)
    return tf.matmul(x, weights) + biases

def conv_layer(inputs, out_num, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, in_chanel, out_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, inputs.get_shape()[-1], out_num],
                          0.02)
    
    biases = get_biases(l_id, [out_num], 0.0)
    
    conved = tf.nn.conv2d(inputs, weights,
                          strides=[1, stride,  stride,  1],
                          padding = 'SAME')
    
    return tf.nn.bias_add(conved, biases)


def deconv_layer(inputs, out_shape, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, out_chanel, in_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, out_shape[-1], inputs.get_shape()[-1]],
                          0.02)
    
    biases = get_biases(l_id, [out_shape[-1]], 0.0)
    
    deconved = tf.nn.conv2d_transpose(inputs, weights, output_shape = out_shape,
                                      strides=[1, stride,  stride,  1])
    return tf.nn.bias_add(deconved, biases)
