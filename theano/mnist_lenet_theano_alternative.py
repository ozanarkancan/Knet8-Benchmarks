# sudo ldconfig /home/ec2-user/cudnn/5.1.3/lib64/ /home/ec2-user/cuda/7.5.18/lib64
# THEANO_FLAGS='floatX=float32,device=gpu0,allow_gc=False,optimizer_including=cudnn,lib.cnmem=1' theano_py mnist_lenet_theano_alternative.py

from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from softmax import SoftmaxRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        relu_out = T.nnet.relu(conv_out, self.b.dimshuffle('x', 0, 'x', 'x'))

        pooled_out = pool.pool_2d(
            input=relu_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = pooled_out
        self.params = [self.W, self.b]
        self.input = input


def evaluate_lenet5(learning_rate=0.1, n_epochs=1,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50, 500], batch_size=100, fast=True):
    rng = numpy.random.RandomState(1)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    print('... building the model')

    layer0_input = x.reshape((batch_size, 1, 28, 28))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=nkerns[2],
        activation=T.nnet.relu
    )

    layer3 = SoftmaxRegression(input=layer2.output, n_in=500, n_out=10)

    cost = layer3.negative_log_likelihood(y)

    valid_loss = theano.function(
        [index],
        cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_loss = theano.function(
        [index],
        cost,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    valid_accuracy = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_accuracy = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training')

    best_valid_loss = numpy.inf
    start_time = timeit.default_timer()
    epoch = 0

    while epoch < n_epochs:
        epoch = epoch + 1
        numloss, sumloss = 0, 0.0
        for minibatch_index in range(n_train_batches):
            cost_ij = train_model(minibatch_index)
            numloss += 1
            sumloss += cost_ij

        if fast:
            continue

        valid_losses = [valid_loss(i) for i in range(n_valid_batches)]
        this_valid_loss = numy.mean(valid_losses)

        if this_valid_loss < best_valid_loss:
            best_valid_loss = this_valid_loss

        print(('     epoch %i, valid loss %f') % (epoch, this_valid_loss))

    end_time = timeit.default_timer()

    lossval = numpy.mean([valid_loss(i) for i in range(n_valid_batches)])
    losstrn = numpy.mean([train_loss(i) for i in range(n_train_batches)])
    valacc = numpy.mean([valid_accuracy(i) for i in range(n_valid_batches)])
    trnacc = numpy.mean([train_accuracy(i) for i in range(n_train_batches)])
    print(('epoch: %d, loss: %f/%f, accuracy: %f/%f in %.4f seconds') %
          (n_epochs, losstrn, lossval, trnacc, valacc, (end_time-start_time)))


if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
