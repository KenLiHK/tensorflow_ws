# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 23:52:59 2017

@author: Ken Li
"""
import tensorflow as tf
import numpy as np

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

#Create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# This function is deprecated and will be removed after 2017-03-02.
#init = tf.initialize_all_variables() 

init = tf.global_variables_initializer()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)   # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
        


