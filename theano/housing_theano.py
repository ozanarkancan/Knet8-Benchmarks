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
import csv

def load_data():
	data = numpy.loadtxt('../data/housing.data', theano.config.floatX)
	
	x = data[:, 0:13]
	y = data[:, 13:]

	x = (x - numpy.mean(x,0)) / numpy.std(x,0)

	print x.shape
	print y.shape
	return (x, y)

def timit_linreg():
	#Parameters
	n_in = 13
	n_out = 1
	lr = 0.1


	print '...loading the data'
	
	trnx, trny = load_data()
	

	print '...building the model'

	index = T.lscalar()
	x = T.matrix('x')
	y = T.matrix('y')

	#Model Parameters

	Wo_values = numpy.asarray(0.1*numpy.random.randn(n_in, n_out),
			dtype=theano.config.floatX
			)
	Wo = theano.shared(value=Wo_values, name='Wo', borrow=True)
	bo_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        bo = theano.shared(value=bo_values, name='bo', borrow=True)

	#Model
	#x* W + b

	ypred = T.dot(x, Wo) + bo
	mse = T.mean((ypred - y) ** 2)

	params = [Wo, bo]
	gparams = [T.grad(mse, param) for param in params]

	updates = [ (param, param - lr * gparam) for param, gparam in zip(params, gparams)]
	
	train_model = theano.function(
		inputs=[],
		outputs=mse,
		updates=updates,
		allow_input_downcast=True,
		givens={
			x: trnx,
			y: trny
		}
		)

	print '...training'

	gc.disable()
	start_time = timer()
	
	for i in range(10000):
		cost = train_model()
	
	end_time = timer()
	
	gc.enable()
	print 'Loss: {}, Time: {} seconds'.format(cost, end_time - start_time)
	
	#print '\nSingle forw+back+update call\n'

	#for i in range(10):
	#	gc.disable()
	#	start_time = timer()
	#	cost = train_model()
	#	end_time = timer()
	#	gc.enable()

	#	print 'Epoch: {}, Loss: {}, Time: {} seconds'.format(i+1, cost, end_time - start_time)


if __name__ == "__main__":
	timit_linreg()
