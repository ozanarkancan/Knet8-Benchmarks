# This implementation is faster compared with the native implementation because the data moved to gpu and removed the feed dict implementation
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import timeit
# Network parameters
n_hidden = 64
n_input = 784
n_classes = 10


mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

batch_xs, batch_ys = mnist.train.next_batch(100)
batch_y = batch_ys.astype(np.float32)

with tf.Graph().as_default():
    with tf.device('/gpu:0'):
        x = tf.constant(batch_xs, name='x')
        y_ = tf.constant(batch_y, name='y')
    # model weights and biases
    W_ih = tf.Variable(0.1 * tf.random_normal([n_input, n_hidden]))
    b_ih = tf.Variable(tf.zeros([n_hidden]))
    W_ho = tf.Variable(0.1 * tf.random_normal([n_hidden, n_classes]))
    b_ho = tf.Variable(tf.zeros([n_classes]))
    # create the model
    hidden_layer =tf.nn.relu (tf.add(tf.matmul(x, W_ih), b_ih))
    logits = tf.add(tf.matmul(hidden_layer, W_ho), b_ho)


    # define loss and train step
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)
    start_time = timeit.default_timer()
    for i in range(5500):
        sess.run(train_step)
    end_time = timeit.default_timer()
    print 'Total time is',  (end_time - start_time)



