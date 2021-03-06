# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 00:15:04 2017

This code is referring to the code from this link:
https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow8_feeds.py

"""

import tensorflow as tf

print('tensorflow version:', tf.__version__)

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)


with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))



