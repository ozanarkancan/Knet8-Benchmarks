import tensorflow as tf
import numpy as np
import timeit

# load the data
data = np.load('datafile.npz')
train_y = data['arr_1']
x = data['arr_0']
x = x.astype(np.float32)
train_y = train_y.astype(np.float32)

# model parameters
epochs = 10000
learning_rate = 0.1

# Model inputs
X = tf.constant(x, name="X")
Y = tf.constant(train_y, name="Y")


# Model parameters
W = tf.Variable(0.01*tf.random_normal([1,13], dtype=tf.float32))
b = tf.Variable(tf.zeros([1],  dtype=tf.float32))

# Construct the model
model = tf.add(tf.matmul(W, X), b)

# Minimize squared errors
cost_function = tf.reduce_sum(tf.pow(model - Y, 2)/ train_y.shape[1])
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function) #Gradient descent


# Initialize variables
init = tf.initialize_all_variables()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    initial_cost = sess.run(cost_function)
    print "initial cost is", initial_cost
    start_time = timeit.default_timer()
    for epoch in range(epochs):
        sess.run(optimizer)
#        current_cost = sess.run(cost_function, feed_dict={X: x ,Y:train_y})
#        print current_cost
    end_time = timeit.default_timer()
    final_cost = sess.run(cost_function)
    print("Total time %.4fs" % (end_time - start_time))
print("Final cost is %4.f" % (final_cost))


# This part of the code is actually done by a different script to load the data namely, x and train_y

# preprocessing
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
#rawdata = np.asarray(pd.read_csv(url, sep='\s+', header=None)) # first 13 columns of the data is X and the last one is Y
#train_x = np.transpose(rawdata[:,0:13]) # each column of the x corresponds to one training example
#train_y = np.reshape(rawdata[:,-1],(1,506)) # each column corresponds to one ygold

#meanx = np.mean(train_x, axis=1)
#meanx = np.reshape(meanx, (meanx.shape[0],1))
#stdx = np.std(train_x, axis=1, ddof=1)
#stdx = np.reshape(stdx, (stdx.shape[0],1))

#x = np.divide((np.subtract(train_x, meanx)), stdx)
