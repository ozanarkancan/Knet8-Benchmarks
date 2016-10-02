import numpy as np
import timeit
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
batch_size = 100
batch_x, batch_ys = mnist.train.next_batch(batch_size)
batch_y = batch_ys.astype(np.float32)

# Parameters
learning_rate = 0.1

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)




# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)


    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Launch the graph
with tf.Graph().as_default():
    with tf.device('/gpu:0'):
        # tf Graph input
        x = tf.constant(batch_x)
        y = tf.constant(batch_y)

    
        # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 20 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 20])),
        # 5x5 conv, 20 inputs, 50 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 20, 50])),
        # fully connected, 7*7*64 inputs, 500 outputs
        'wd1': tf.Variable(tf.random_normal([7 * 7* 50, 500])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([500, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([20])),
        'bc2': tf.Variable(tf.random_normal([50])),
        'bd1': tf.Variable(tf.random_normal([500])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = conv_net(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


    # Initializing the variables
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    sess = tf.Session()  

    sess.run(init_op)
    

    # Run optimization op (backprop)
    start_time = timeit.default_timer()
    for i in range(550):
        sess.run(optimizer)
    end_time = timeit.default_timer()
    print ("Total time is ", (end_time - start_time))
