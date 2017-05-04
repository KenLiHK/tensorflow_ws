# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:37:37 2017

Created by: Ken Li
"""

import tensorflow as tf

print('*** cmf_ts19.py ***')
print('tensorflow version:', tf.__version__)

## Save to file
# should define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32,name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, 'my_net/save_net.ckpt')
    print("Save to path:", save_path)





