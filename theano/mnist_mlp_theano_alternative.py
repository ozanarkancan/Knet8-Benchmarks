# sudo ldconfig /home/ec2-user/cudnn/5.1.3/lib64/ /home/ec2-user/cuda/7.5.18/lib64
# THEANO_FLAGS='floatX=float32,device=gpu0,allow_gc=False,optimizer_including=cudnn,lib.cnmem=1' theano_py mnist_mlp_theano_alternative.py

from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from softmax import SoftmaxRegression, load_data


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.relu):
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = SoftmaxRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input


def test_mlp(learning_rate=0.5, L1_reg=0.00, L2_reg=0.0000, n_epochs=10,
             dataset='mnist.pkl.gz', batch_size=100, n_hidden=64, fast=True):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    cost = classifier.negative_log_likelihood(y)

    valid_loss = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    train_loss = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    valid_accuracy = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    train_accuracy = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
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
            minibatch_avg_cost = train_model(minibatch_index)
            numloss += 1
            sumloss += minibatch_avg_cost

        if fast:
            continue

        valid_losses = [valid_loss(i) for i in range(n_valid_batches)]
        this_valid_loss = numpy.mean(valid_losses)

        if this_valid_loss < best_valid_loss:
            best_valid_loss = this_valid_loss
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(classifier, f)

    end_time = timeit.default_timer()

    lossval = numpy.mean([valid_loss(i) for i in range(n_valid_batches)])
    losstrn = numpy.mean([train_loss(i) for i in range(n_train_batches)])
    valacc = numpy.mean([valid_accuracy(i) for i in range(n_valid_batches)])
    trnacc = numpy.mean([train_accuracy(i) for i in range(n_train_batches)])
    print(('epoch: %d, loss: %f/%f, accuracy: %f/%f in %.4f seconds') %
          (n_epochs, losstrn, lossval, trnacc, valacc, (end_time-start_time)))


if __name__ == '__main__':
    print('floatX: %s, device: %s' % (theano.config.floatX, theano.config.device))
    test_mlp()
