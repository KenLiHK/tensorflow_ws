# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 00:07:23 2017

This code is referring to the code from this link:
https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow7_variable.py

"""

import tensorflow as tf

print('tensorflow version:', tf.__version__)

state = tf.Variable(0, name='counter')
print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer() # must have if defined variable


with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

