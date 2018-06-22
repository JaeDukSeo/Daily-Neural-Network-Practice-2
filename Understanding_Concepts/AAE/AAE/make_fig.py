#! -*- coding:utf-8 -*-

import os
import sys

def zero_or_not(target):
    #return 1 if target > 0 else 0
    return target/127.5 - 1.0

def line2label_fig(l):
    #zeros = [0.0] * 10
    zeros = [0.0] * 11
    tmp = l.strip().split(",")
    zeros[int(tmp[0])] = 1.0
    fig = [zero_or_not(int(i)) for i in tmp[1:]]
    #tmp2 = [[zero_or_not(int(i))] for i in tmp[1:]]
    #fig = [tmp2[28 * i : 28 * i + 28] for i in range(28)]
    return zeros, fig

def get_batch(f_obj, num_batch):
    label_ret = []
    fig_ret = []
    for i, l in enumerate(f_obj):
        label, fig = line2label_fig(l)
        label_ret.append(label)
        fig_ret.append(fig)
        if i == num_batch - 1:
            break
    return label_ret, fig_ret

if __name__ == u'__main__':
    with open('mnist_test.csv', 'r') as f:
        labels, figs = get_batch(f, 2)
        print(figs)
        
