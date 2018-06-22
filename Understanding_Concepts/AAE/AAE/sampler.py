#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import math

import matplotlib.pyplot as plt

class Sampler(object):
    def __init__(self, class_num):
        # 0 -- class_num - 1: calssification index
        # class_num: for unlabeled index
        self.class_num = class_num

        self.x_variance = 0.5
        self.y_variance = 0.05
        self.radial = 2.0
        
    def _get_radian(self, class_index):
        return 2 * np.pi * float(class_index)/float(self.class_num)

    def _rotate(self, x, y, radian):
        mod_x = x * math.cos(radian) - y * math.sin(radian)
        mod_y = x * math.sin(radian) + y * math.cos(radian)
        return mod_x, mod_y
    
    def __call__(self, class_indexes):
        ret = []
        for class_index in class_indexes:
            x = np.random.normal(0.0, self.x_variance) + self.radial
            y = np.random.normal(0.0, self.y_variance)
            rad = self._get_radian(class_index)
            x, y = self._rotate(x, y, rad)
            ret.append([x, y])
            
        return np.asarray(ret)


if __name__ == u'__main__':
    s = Sampler(10)
    tmp = [[0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 1]]
    tmptmp = np.argmax(tmp, axis = 1)
    p = s(tmptmp)
    x = p[:,0]
    y = p[:,1]
    plt.scatter(x, y)
    plt.show()
