# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:38:05 2017

This code is referring to the code from this link:
https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf20_RNN2/full_code.py

"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

print('*** cmf_ts20.py ***')
print('*** 20.1 - 20.4 ***')
print('tensorflow version:', tf.__version__)

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28 #MNIST data input (img shape: 28*28)
n_steps = 28 #time steps
n_hidden_unis = 128 #neurons in hidden layer
n_classes = 10 #MNIST classes (0-9 digits)

#tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
卜 = tf.placeholder(tf.float32, [None, n_classes])

#Define weights
weights = {
    #(28, 128)
    'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
    #(128, 10)
    'out':tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
}
biases = {
    #(128, )
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_unis, ])),
    #(10, )
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}






