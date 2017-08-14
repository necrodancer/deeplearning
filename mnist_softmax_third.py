# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:40:27 2017

@author: cho
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape, initializer=tf.random_normal_initializer())
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v


def main(_):
  #set variables
  iteration = 1000 #for loop number
  learn_rate = 0.5 #learning rate
  batch_size = 10 #batch size
  
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W1 = get_scope_variable('foo','W1',[784,512])
  b1 = get_scope_variable('foo','b1',[512,])
  a1 = tf.matmul(x, W1) + b1
  z1 = tf.nn.sigmoid(a1)
  W2 = get_scope_variable('foo','W2',[512,256])
  b2 = get_scope_variable('foo','B2',[256,])
  a2 = tf.matmul(z1, W2) + b2
  z2 = tf.nn.sigmoid(a2)
  W3 = get_scope_variable('foo','W3',[256,10])
  b3 = get_scope_variable('foo','b3',[10,]) 
  y = tf.matmul(z2,W3) + b3

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(iteration):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)