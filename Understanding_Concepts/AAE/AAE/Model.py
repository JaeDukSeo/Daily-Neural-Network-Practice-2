#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from sampler import Sampler

class Model(object):
    def __init__(self, input_dim, z_dim, class_num, batch_size):
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.class_num = class_num
        
        self.batch_size = batch_size
        self.lr = 0.0001
        
        # -- encoder -------
        self.encoder = Encoder([input_dim, 1200, 600, 100], z_dim)
        
        # -- decoder -------
        self.decoder = Decoder([z_dim, 100, 600, 1200, input_dim])

        # -- discriminator --
        self.discriminator = Discriminator([z_dim + (class_num + 1), 50, 20, 10, 1])
        
        # -- sampler ----
        self.sampler = Sampler(class_num)
        
    def set_model(self):
        # TODO: only labeled
        
        # -- for labeled data ------
        self.x_labeled = tf.placeholder(tf.float32, [self.batch_size, self.input_dim])

        # encode and decode
        mu, log_sigma = self.encoder.set_model(self.x_labeled, is_training = True)
                
        eps = tf.random_normal([self.batch_size, self.z_dim])
        z = eps * tf.exp(log_sigma) + mu

        gen_figs = self.decoder.set_model(z, is_training = True)

        reconstruct_error = tf.reduce_mean(
            tf.reduce_sum(tf.pow(gen_figs - self.x_labeled, 2), [1]))
        
        # make GAN loss
        self.y_labeled = tf.placeholder(tf.float32, [self.batch_size, self.class_num + 1])
        
        vae_logits = self.discriminator.set_model(z, self.y_labeled, is_training = True, reuse = False)
        obj_disc_from_vae = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = vae_logits,
                labels = tf.zeros_like(vae_logits)))
        obj_gen_from_vae = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = vae_logits,
                labels = tf.ones_like(vae_logits)))

        # discriminator
        self.z_input = tf.placeholder(dtype = tf.float32, shape = [self.batch_size, self.z_dim])
        disc_logits = self.discriminator.set_model(self.z_input, self.y_labeled, is_training = True, reuse = True)
        obj_disc_from_inputs = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = disc_logits,
                labels = tf.ones_like(disc_logits)))

        # -- train -----
        self.obj_vae = reconstruct_error
        train_vars = self.encoder.get_variables()
        train_vars.extend(self.decoder.get_variables())
        self.train_vae  = tf.train.AdamOptimizer(self.lr).minimize(self.obj_vae, var_list = train_vars)
        self.obj_gen = obj_gen_from_vae
        train_vars = self.encoder.get_variables()
        self.train_gen  = tf.train.AdamOptimizer(self.lr).minimize(self.obj_gen, var_list = train_vars)
        self.obj_disc = obj_disc_from_vae + obj_disc_from_inputs
        train_vars = self.discriminator.get_variables()
        self.train_disc  = tf.train.AdamOptimizer(self.lr).minimize(self.obj_disc, var_list = train_vars)
        # -- for using ---------------------
        self.mu, _  = self.encoder.set_model(self.x_labeled, is_training = False, reuse = True)
        self.generate_figs = self.decoder.set_model(self.z_input, is_training = False, reuse = True)
        
    def training_vae(self, sess, figs):
        _, obj_vae = sess.run([self.train_vae, self.obj_vae],
                                  feed_dict = {self.x_labeled: figs})
        return obj_vae
        
    def training_gen(self, sess, figs, y):
        _, obj_gen = sess.run([self.train_gen, self.obj_gen],
                                  feed_dict = {self.x_labeled: figs,
                                               self.y_labeled: y})
        return obj_gen
    
    def training_disc(self, sess, figs, y):
        tmp = np.argmax(y, axis = 1)
        z = self.sampler(tmp)
        _, obj_disc = sess.run([self.train_disc, self.obj_disc],
                                  feed_dict = {self.x_labeled: figs,
                                               self.y_labeled: y,
                                               self.z_input:z})
        return obj_disc
    
    def encoding(self, sess, figs):
        ret = sess.run(self.mu, feed_dict = {self.x_labeled: figs})
        return ret
    def figure_generate(self, sess, z):
        figs = sess.run(self.generate_figs, feed_dict = {self.z_input: z})
        return figs
    
if __name__ == u'__main__':
    model = Model(28 * 28 * 1, 2, 10, 100)
    model.set_model()
    
