"""

Code is based on Theano Deep Learning Tutorials

"""

import six.moves.cPickle as pickle
import gzip
import os
import sys
from timeit import default_timer as timer

import numpy

import theano
import theano.tensor as T
import gc
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

def load_data(dataset):
	with gzip.open(dataset, 'rb') as f:
		try:
			train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
		except:
			train_set, valid_set, test_set = pickle.load(f)

	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x,
			dtype=theano.config.floatX),borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
			dtype=theano.config.floatX), borrow=borrow)
		return shared_x, T.cast(shared_y, 'int32')
	
	trnx = numpy.vstack((train_set[0], valid_set[0]))
	trny = numpy.concatenate((train_set[1], valid_set[1]))
	
	train_set_x, train_set_y = shared_dataset((trnx, trny))
	
	return (train_set_x, train_set_y)

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def timit_lenet():
	#Parameters
	n_in = 28*28
	n_hidden = 500
	n_out = 10
	batch_size = 100
	lr = 0.1

	dataset = '../data/mnist.pkl.gz'

	print '...loading the data'
	
	train_set_x, train_set_y = load_data(dataset)
	
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
	#n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
	#n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

	rng = numpy.random.RandomState(1234)

	print '...building the model'

	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	layer0_input = x.reshape((batch_size, 1, 28, 28))
	
	layer0 = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 28, 28),
			filter_shape=(20, 1, 5, 5), poolsize=(2, 2))
	layer1 = LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, 20, 12, 12),
			filter_shape=(50, 20, 5, 5), poolsize=(2, 2))

	layer2_input = layer1.output.flatten(2)

	#Model Parameters

        W_bound = numpy.sqrt(6. / (800 + n_hidden))
	Wh_values = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(800, 500)),
			dtype=theano.config.floatX
            )
	
	Wh = theano.shared(value=Wh_values, name='Wh', borrow=True)
	bh_values = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
        bh = theano.shared(value=bh_values, name='bh', borrow=True)

	W_bound = numpy.sqrt(6. / (500 + 10))
	Wo_values = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(500, 10)),
			dtype=theano.config.floatX
            )

	Wo = theano.shared(value=Wo_values, name='Wo', borrow=True)
	bo_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        bo = theano.shared(value=bo_values, name='bo', borrow=True)

	hidden = T.nnet.relu(T.dot(layer2_input, Wh) + bh)
	p_y_given_x = T.nnet.softmax(T.dot(hidden, Wo) + bo)

	nll = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

	params = layer0.params + layer1.params + [Wh, bh, Wo, bo]
	gparams = [T.grad(nll, param) for param in params]

	updates = [ (param, param - lr * gparam) for param, gparam in zip(params, gparams)]
	
	train_model = theano.function(
		inputs=[index],
		outputs=nll,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)

	print '...training'

	gc.disable()
	start_time = timer()
	
	for i in range(1):
		total_cost = 0.0
		count = 0.0
		for minibatch_index in range(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
			total_cost += minibatch_avg_cost
			count += 1
	
	end_time = timer()
	
	gc.enable()
	print 'Loss: {}, Time: {} seconds'.format(total_cost/count, end_time - start_time)
	
	print '\nSingle forw+back+update call\n'

	for i in range(10):
		gc.disable()
		start_time = timer()
		minibatch_avg_cost = train_model(i)
		end_time = timer()
		gc.enable()

		print 'Epoch: {}, Loss(Batch): {}, Time: {} seconds'.format(i+1, minibatch_avg_cost, end_time - start_time)


if __name__ == "__main__":
	timit_lenet()
