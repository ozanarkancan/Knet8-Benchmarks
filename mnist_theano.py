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
	
	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)
	
	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
	(test_set_x, test_set_y)]
	
	return rval


def timit_mlp():
	#Parameters
	n_in = 28*28
	n_hidden = 100
	n_out = 10
	batch_size = 100
	lr = 0.01

	dataset = 'data/mnist.pkl.gz'

	print '...loading the data'

	datasets = load_data(dataset)
	
	train_set_x, train_set_y = datasets[0]
	#valid_set_x, valid_set_y = datasets[1]
	#test_set_x, test_set_y = datasets[2]
	
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
	#n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
	#n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

	rng = numpy.random.RandomState(1234)

	print '...building the model'

	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	#Model Parameters

	Wh_values = numpy.asarray(
			rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_hidden)),
				high=numpy.sqrt(6. / (n_in + n_hidden)),
				size=(n_in, n_hidden)
				),
			dtype=theano.config.floatX
			)
	Wh = theano.shared(value=Wh_values, name='Wh', borrow=True)
	bh_values = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
        bh = theano.shared(value=bh_values, name='bh', borrow=True)

	Wo_values = numpy.asarray(
			rng.uniform(
				low=-numpy.sqrt(6. / (n_hidden + n_out)),
				high=numpy.sqrt(6. / (n_hidden + n_out)),
				size=(n_hidden, n_out)
				),
			dtype=theano.config.floatX
			)
	Wo = theano.shared(value=Wo_values, name='Wo', borrow=True)
	bo_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        bo = theano.shared(value=bo_values, name='bo', borrow=True)

	#Model
	#mlp with 64 hidden units, activation function is relu

	hidden = T.nnet.relu(T.dot(x, Wh) + bh)
	p_y_given_x = T.nnet.softmax(T.dot(hidden, Wo) + bo)

	nll = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

	params = [Wh, bh, Wo, bo]
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

	for i in range(10):
		total_cost = 0.0
		count = 0.0
		start_time = timer()
		for minibatch_index in range(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
			total_cost += minibatch_avg_cost
			count += 1
		end_time = timer()
		print 'Epoch: {}, Loss: {}, Time: {} seconds'.format(i+1, total_cost/count, end_time - start_time)
	
	print '\nSingle forw+back+update call\n'

	for i in range(10):
		start_time = timer()
		minibatch_avg_cost = train_model(i)
		end_time = timer()

		print 'Epoch: {}, Loss(Batch): {}, Time: {} seconds'.format(i+1, minibatch_avg_cost, end_time - start_time)


if __name__ == "__main__":
	timit_mlp()