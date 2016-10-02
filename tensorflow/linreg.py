import tensorflow as tf
import numpy as np
import timeit

data = np.load('datafile.npz')
train_y = data['arr_1']


x = data['arr_0']
x = x.astype(np.float32)
train_y = train_y.astype(np.float32)

# model parameters
epochs = 10000
learning_rate = 0.1

with tf.Graph().as_default():
    with tf.device('gpu:0'):
        X = tf.constant(x, name="X")
        Y = tf.constant(train_y, name="Y")

    W = tf.Variable(0.01*tf.random_normal([1,13], dtype=tf.float32))
    b = tf.Variable(tf.zeros([1],  dtype=tf.float32))

    
    # Construct the model
    model = tf.add(tf.matmul(W, X), b)

    # Minimize squared errors
    cost_function = tf.reduce_sum(tf.pow(model - Y, 2)/ 506)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function) #Gradient descent

    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    sess = tf.Session()
    sess.run(init_op)
    initial_cost = sess.run(cost_function)
    print "Initial cost is", initial_cost
    start_time = timeit.default_timer()

    for epoch in range(epochs):
        sess.run(optimizer)
    end_time = timeit.default_timer()
    final_cost = sess.run(cost_function)
    print("Total time %.4fs" % (end_time - start_time))
print("Final cost is %4.f" % (final_cost))

