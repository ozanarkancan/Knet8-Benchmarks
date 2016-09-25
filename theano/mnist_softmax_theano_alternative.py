# sudo ldconfig /home/ec2-user/cudnn/5.1.3/lib64/ /home/ec2-user/cuda/7.5.18/lib64
# THEANO_FLAGS='floatX=float32,device=gpu0,allow_gc=False,optimizer_including=cudnn,lib.cnmem=1' theano_py mnist_softmax_theano_alternative.py

from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy
import gc

import theano
import theano.tensor as T


class SoftmaxRegression(object):
    def __init__(self, input, n_in, n_out, seed=1):
        numpy.random.seed(seed)
        self.W = theano.shared(
            value=numpy.asarray(0.1*numpy.random.randn(n_in, n_out),
            dtype=theano.config.floatX),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    train_set = (
        numpy.concatenate((train_set[0], valid_set[0])),
        numpy.concatenate((train_set[1], valid_set[1])),
    )

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y)]

    return rval


def sgd_optimization_mnist(learning_rate=0.5, n_epochs=10, dataset='mnist.pkl.gz', batch_size=100, fast=True):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

    print('    n_train_batches: %d, n_valid_batches: %d' % (n_train_batches, n_valid_batches))

    print('... building the model')

    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    classifier = SoftmaxRegression(input=x, n_in=28 * 28, n_out=10)
    cost = classifier.negative_log_likelihood(y)

    valid_loss = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_loss = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    valid_accuracy = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_accuracy = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training the model')

    best_valid_loss = numpy.inf
    epoch = 0
    gc.disable()
    start_time = timeit.default_timer()
    
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

        print(('     epoch %i, valid loss %f') % (epoch, valid_score))

    end_time = timeit.default_timer()
    gc.enable()
    lossval = numpy.mean([valid_loss(i) for i in range(n_valid_batches)])
    losstrn = numpy.mean([train_loss(i) for i in range(n_train_batches)])
    valacc = numpy.mean([valid_accuracy(i) for i in range(n_valid_batches)])
    trnacc = numpy.mean([train_accuracy(i) for i in range(n_train_batches)])
    print(('epoch: %d, loss: %f/%f, accuracy: %f/%f in %.4f seconds') %
          (n_epochs, losstrn, lossval, trnacc, valacc, (end_time-start_time)))


def predict():
    classifier = pickle.load(open('best_model.pkl'))

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    valid_set_x, valid_set_y = datasets[1]
    valid_set_x = valid_set_x.get_value()

    predicted_values = predict_model(valid_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':
    print('floatX: %s, device: %s' % (theano.config.floatX, theano.config.device))
    sgd_optimization_mnist()
