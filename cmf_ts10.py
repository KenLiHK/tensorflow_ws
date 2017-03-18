# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:24:38 2017

Created by: Ken Li
"""

import tensorflow as tf
import numpy as np

print('tensorflow version:', tf.__version__)


def add_layer(inputs, in_size, out_size, activation_functino=None):
    Weights = tf.Variable(tf.rnadom_nomal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
    
    










