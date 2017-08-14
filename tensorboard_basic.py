# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 08:40:12 2017

@author: cho
"""
from __future__ import print_function

'''
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example'

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

#define get_scope_variable()
def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape, initializer=tf.random_normal_initializer())
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v

# Set model weights
W1 = get_scope_variable('Weights','W1',[784, 512])
b1 = get_scope_variable('Weights','b1',[512,])
W2 = get_scope_variable('Weights','W2',[512, 256])
b2 = get_scope_variable('Weights','b2',[256,])
W3 = get_scope_variable('Weights','W3',[256, 10])
b3 = get_scope_variable('Weights','b3',[10,])
'''
with tf.name_scope('Weights'):    
    W1 = tf.Variable(tf.zeros([784, 512]), name='Weights1')
    b1 = tf.Variable(tf.zeros([512]), name='Bias1')
    W2 = tf.Variable(tf.zeros([512, 256]), name='Weights2')
    b2 = tf.Variable(tf.zeros([256]), name='Bias2')
    W3 = tf.Variable(tf.zeros([256, 10]), name='Weights3')
    b3 = tf.Variable(tf.zeros([10]), name='Bias3')
'''

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    z1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1) # layer 0
    z2 = tf.nn.sigmoid(tf.matmul(z1, W2) + b2) # layer 1
    pred = tf.nn.softmax(tf.matmul(z2, W3) + b3) # layer 2 ;Softmax
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
"\nThen open http://0.0.0.0:6006/ into your web browser")
