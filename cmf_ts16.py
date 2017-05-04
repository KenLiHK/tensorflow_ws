# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 05:43:29 2017

This code is referring to the code from this link:
https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf16_classification/full_code.py

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

print('*** cmf_ts16.py ***')
print('tensorflow version:', tf.__version__)


# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram('weights', Weights)
        with tf.name_scope('biases'):        
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram('biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram('outputs', outputs)
        return outputs
    

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

# Make up some real data    
#x_data = np.linspace(-1,1,300)[:,np.newaxis]
#noise = np.random.normal(0,0.05,x_data.shape)
#y_data = np.square(x_data) - 0.5 + noise


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784], name='x_input') # 28 x 28
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
    

# add hidden layer    
#l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)


# add output layer
#prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediciton and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1])) #loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
    
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
'''

init = tf.global_variables_initializer() # must have if defined variable
sess = tf.Session()


#collect all log from session and put into logs folder 
#for displaying in visualization tool
#merged = tf.merge_all_summaries()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
"""

for i in range(1000):
    #sess.run(train_step,feed_dict={xs:x_data, ys:y_data})
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
        #to see the step improvement
        #print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        #result = sess.run(merged,feed_dict={xs:x_data, ys:y_data})
        #writer.add_summary(result, i)
        """
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
        """
        
        
