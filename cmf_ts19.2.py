# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:19:48 2017

Created by: Ken Li
"""

import tensorflow as tf
import numpy as np

print('*** cmf_ts19.2.py ***')
print('tensorflow version:', tf.__version__)

# restore variables
# redefine the same shape and same type for yoru variables
W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name="biases")

# no need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))
