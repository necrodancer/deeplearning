# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:37:25 2017

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


def main(_):
  #set variables
  iteration = 1000 #for loop number
  learn_rate = 0.7 #learning rate
  batch_size = 200 #batch size
  
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W1 = tf.Variable(tf.random_normal([784,1024]))
  b1 = tf.Variable(tf.random_normal([1024,]))
  a1 = tf.matmul(x, W1) + b1
  z1 = tf.nn.sigmoid(a1)
  W2 = tf.Variable(tf.random_normal([1024,512]))
  b2 = tf.Variable(tf.random_normal([512,]))
  a2 = tf.matmul(z1, W2) + b2
  z2 = tf.nn.sigmoid(a2)
  W3 = tf.Variable(tf.random_normal([512,256]))
  b3 = tf.Variable(tf.random_normal([256,]))
  a3 = tf.matmul(z2,W3) + b3
  z3 = tf.nn.sigmoid(a3)
  W4 = tf.Variable(tf.random_normal([256,128]))
  b4 = tf.Variable(tf.random_normal([128,]))
  a4 = tf.matmul(z3,W4) + b4
  z4 = tf.nn.sigmoid(a4)
  W5 = tf.Variable(tf.random_normal([128,64]))
  b5 = tf.Variable(tf.random_normal([64,]))
  a5 = tf.matmul(z4,W5) + b5
  z5 = tf.nn.sigmoid(a5)
  W6 = tf.Variable(tf.random_normal([64,512]))
  b6 = tf.Variable(tf.random_normal([512,]))
  a6 = tf.matmul(z5,W6) + b6
  z6 = tf.nn.sigmoid(a6) 
  W7 = tf.Variable(tf.random_normal([512,256]))
  b7 = tf.Variable(tf.random_normal([256,]))
  a7 = tf.matmul(z6,W7) + b7
  z7 = tf.nn.sigmoid(a7)
  W8 = tf.Variable(tf.random_normal([256,128]))
  b8 = tf.Variable(tf.random_normal([128,]))
  a8 = tf.matmul(z7,W8) + b8
  z8 = tf.nn.sigmoid(a8)
  W9 = tf.Variable(tf.random_normal([128,64]))
  b9 = tf.Variable(tf.random_normal([64,]))
  a9 = tf.matmul(z8,W9) + b9
  z9 = tf.nn.sigmoid(a9)
  W10 = tf.Variable(tf.random_normal([64,10]))
  b10 = tf.Variable(tf.random_normal([10,]))
  y = tf.matmul(z9,W10) + b10

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
