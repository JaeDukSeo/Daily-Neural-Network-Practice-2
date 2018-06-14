#################################################################
# Implementation of Integrated Gradients function in Tensorflow #
# Naozumi Hiranuma (hiranumn@cs.washington.edu)                 #
#################################################################

import tensorflow as tf
import numpy as np

# INPUT: tensor of samples to explain
# OUTPUT: interpolated: linearly interpolated samples between input samples and references.
#         stepsize: stepsizes between samples and references
#         reference: a placeholder tensor for optionally specifying reference values.

def linear_inpterpolation(sample, num_steps=50):
    
    # Constrtuct reference values if not available.
    reference = tf.placeholder_with_default(tf.zeros_like(sample), shape=sample.get_shape())
    
    # Expand sample and reference 
    sample_ = tf.stack([sample for _ in range(num_steps)])
    reference_ = tf.stack([reference for _ in range(num_steps)])
    
    # Get difference between sample and reference
    dif = sample_ - reference_ 
    stepsize = tf.divide(dif, num_steps)
    
    # Get multipliers
    multiplier = tf.divide(tf.stack([tf.ones_like(sample)*i for i in range(num_steps)]), num_steps)
    interploated_dif = tf.multiply(dif, multiplier)
    
    # Get parameters for reshaping
    _shape = [-1] + [int(s) for s in sample.get_shape()[1:]]
    perm = [1, 0]+[i for i in range(2,len(sample_.get_shape()))]
    # Reshape
    interploated = tf.reshape(reference_ + interploated_dif, shape=_shape)
    stepsize = tf.reshape(stepsize, shape=_shape)
    
    return interploated, stepsize, reference

# INPUT: samples: linearly interpolated samples between input samples and references. output of linear_interpolation()
#        stepsizse: output of linear_interpolation()
#        _output: output tensor to be explained. It needs to be connected to samples.
# OUTPUT: explanations: A list of tensors with explanation values. 

def build_ig(samples, stepsizes, _output, num_steps=50):
    grads = tf.gradients(ys=_output, xs=samples)
    
    flag = False
    
    if not isinstance(samples, list):
        samples = [samples]
        stepsizes = [stepsizes]
        flag=True
    
    # Estimate riemann sum
    output = []
    for i in range(len(samples)):
        s = stepsizes[i]
        g = grads[i]
        riemann = tf.multiply(s, g)
        riemann = tf.reshape(riemann, shape=[num_steps,-1]+[int(s) for s in s.get_shape()[1:]])
        explanation = tf.reduce_sum(riemann, axis=0)
        output.append(explanation)
    
    # Return the values. 
    if flag:
        return output[0]
    else:
        return output
