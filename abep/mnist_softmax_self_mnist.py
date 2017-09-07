# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 20:41:14 2017

@author: cho
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys, os
sys.path.append(os.pardir)

import tensorflow as tf
import numpy as np

#import MNIST data
from dataset.mnist import load_mnist
#from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

#define get_scope_variable()
def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape, initializer=tf.random_normal_initializer())
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v

#main
def main(_):
  #parameters
  iteration = 1000 #for loop number
  learn_rate = 0.5 #learning rate
  batch_size = 100 #batch size
  logs_path = '/tmp/tensorflow_logs/example'
  
  # Load MNIST data
  (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=False,one_hot_label=True)
  #(mnist_train_images,mnist_train_label),(mnist_test_images,mnist_test_label) = input_data.read_mnist()

  # tf Graph INput
  # MNIST data image of shape 28*28=784
  x = tf.placeholder(tf.float32, [None, 784])
  # Set model weights and output for layer 0
  W1 = get_scope_variable('foo','W1',[784,512])
  b1 = get_scope_variable('foo','b1',[512,])
  a1 = tf.matmul(x, W1) + b1
  z1 = tf.nn.sigmoid(a1)
  # Set model weights and output for layer 1
  W2 = get_scope_variable('foo','W2',[512,256])
  b2 = get_scope_variable('foo','B2',[256,])
  a2 = tf.matmul(z1, W2) + b2
  z2 = tf.nn.sigmoid(a2) 
  # Set model weights and output for layer 2
  W3 = get_scope_variable('foo','W3',[256,10])
  b3 = get_scope_variable('foo','b3',[10,])   
  y = tf.matmul(z2,W3) + b3  
  # 0-9 digits recognition => 10 classes
  y_ = tf.placeholder(tf.float32, [None, 10])  
  
  # Define loss and optimizer
  # Minimize error using cross entropy
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
  
  # Create a summary to monitor cost tensor
  tf.summary.scalar("loss", cross_entropy)
  # Merge all summaries into a single op
  #merged_summary_op = tf.summary.merge_all() 
  
   # Initializing the variables and launch the graph.
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  # op to write logs to Tensorboard
  summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
  

  # Train cycle
  for i in range(iteration):
    train_size = x_train.shape[0]
    batch_mask = np.random.choice(train_size,batch_size)
    batch_xs = x_train[batch_mask]
    batch_ys = t_train[batch_mask]
    #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # Write logs at every iteration
    summary_writer.add_summary(sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}), batch_size + i)

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: x_test,
                                      y_: t_test}))
                                      
# Exception for executing main
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
