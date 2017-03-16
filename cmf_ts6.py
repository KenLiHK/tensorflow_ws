# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 23:52:59 2017

Created by: Ken Li
"""

import tensorflow as tf

print('tensorflow version:', tf.__version__)

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m2, m2)

# method 1
sess = tf.Session()
result = sess.run(product)
print('method1', result)
sess.close()

# method 2 session will be close after with section
with tf.Session() as sess:
    result2 = sess.run(product)
    print('method2', result2)