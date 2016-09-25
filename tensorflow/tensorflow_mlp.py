# MLP implementation in tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import timeit

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# Parameters
learning_rate = 0.5
batch_size = 100

# Network parameters
n_hidden = 64
n_input = 784
n_classes = 10

# tf graph input
x = tf.placeholder(tf.float32, [None, n_input])
ygold = tf.placeholder(tf.float32, [None, n_classes])

# model weights and biases
W_ih = tf.Variable(0.1 * tf.random_normal([n_input, n_hidden]))
b_ih = tf.Variable(tf.zeros([n_hidden]))
W_ho = tf.Variable(0.1 * tf.random_normal([n_hidden, n_classes]))
b_ho = tf.Variable(tf.zeros([n_classes]))

# create the model
hidden_layer = tf.add(tf.matmul(x, W_ih), b_ih)
hidden_layer = tf.nn.relu(hidden_layer)
out_layer = tf.add(tf.matmul(hidden_layer, W_ho), b_ho)


# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out_layer, ygold))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    start_time = timeit.default_timer()
    while mnist.train.epochs_completed != 10:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, ygold:batch_ys})
    end_time = timeit.default_timer()
    print("Total time is %s" % (end_time - start_time))
# Test trained model This part agrees with the official documentation so the implementation is correct
    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(ygold, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Final accuracy is", (accuracy.eval({x: mnist.test.images, ygold: mnist.test.labels})))

