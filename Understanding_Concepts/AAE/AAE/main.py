#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import argparse

from Model import Model
#from util import get_figs, dump_figs
from make_fig import get_batch

if __name__ == u'__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', default = None, type = str)
    args = parser.parse_args()
    
    # parameter
    batch_size = 100
    epoch_num = 100
    z_dim = 2
    
    # make model
    print('-- make model --')
    model = Model(28 * 28 * 1, z_dim, 10, batch_size)
    model.set_model()
    
    # get_data
    print('-- get figs--')
    #with open('mnist_test.csv') as f:
    with open(args.data) as f:
        labels, figs = get_batch(f, 5000)
    print('num figs = {}'.format(len(figs)))
    
    # training
    print('-- begin training --')
    num_one_epoch = len(figs) //batch_size

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epoch_num):

            print('** epoch {} begin **'.format(epoch))
            obj_vae, obj_dec, obj_disc = 0.0, 0.0, 0.0
            for step in range(num_one_epoch):
                
                # get batch data
                batch_figs = figs[step * batch_size: (step + 1) * batch_size]
                batch_labels = labels[step * batch_size: (step + 1) * batch_size]

                # train
                obj_disc += model.training_disc(sess, batch_figs, batch_labels)
                obj_vae += model.training_vae(sess, batch_figs)
                model.training_gen(sess, batch_figs, batch_labels)
                model.training_gen(sess, batch_figs, batch_labels)
                model.training_gen(sess, batch_figs, batch_labels)
                model.training_gen(sess, batch_figs, batch_labels)
                model.training_gen(sess, batch_figs, batch_labels)
                obj_dec += model.training_gen(sess, batch_figs, batch_labels)
                
                if step%10 == 0:
                    print('   step {}/{} end'.format(step, num_one_epoch));sys.stdout.flush()
                    
            print('epoch:{}, v_obj = {}, dec_obj = {}, disc_obj = {}'.format(epoch,
                                                                        obj_vae/num_one_epoch,
                                                            obj_dec/num_one_epoch,
                                                            obj_disc/num_one_epoch))
            saver.save(sess, './model.dump')
