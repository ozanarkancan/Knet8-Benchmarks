# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import timeit

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

batch_xs, batch_ys = mnist.train.next_batch(100)


# Create the model
x = tf.constant(batch_xs, name="x")
W = tf.Variable(0.1*tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
logits = tf.matmul(x, W) + b

# Define loss and optimizer, take only one single batch make forward, backward and update on it
batch_y = batch_ys.astype(np.float32)
y_ = tf.constant(batch_y, name="y_")
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  

# Train with single minibatch
init = tf.initialize_all_variables()

with tf.Session() as sess:
  sess.run(init)
  initial_loss = sess.run(loss)
  print('Initial loss is ', initial_loss)
  start_time = timeit.default_timer()
  for i in range(5500):
    sess.run(train_step)
  end_time = timeit.default_timer()
  final_loss = sess.run(loss)
  print("Final loss is", final_loss)

print("Total time is %s" % (end_time - start_time))

# Test trained model This part agrees with the official documentation so the implementation is correct
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

# Naive implementation that is up to 2x slower
# Train
#tf.initialize_all_variables().run()
#start_time = timeit.default_timer()
#while mnist.train.epochs_completed != 10:
#  batch_xs, batch_ys = mnist.train.next_batch(100)
#  train_step.run({x: batch_xs, y_: batch_ys})
#end_time = timeit.default_timer()


