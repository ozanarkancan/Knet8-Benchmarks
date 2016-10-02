#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T

from softmax import *
from lstm import *
from logistic import *
from updates import *

class RNN(object):
    def __init__(self, in_size, out_size, hidden_size, XX, YY,
                 cell = "gru", optimizer = "rmsprop"):
	self.Xdata = XX
	self.Ydata = YY
        self.X = T.tensor3("X")
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.index = T.iscalar('index')
        self.batch_size = T.iscalar('batch_size') # for mini-batch training
        self.optimizer = optimizer
        self.define_layers()
        self.define_train_test_funcs()
        
    def define_layers(self):
        self.layers = []
        self.params = []
        rng = np.random.RandomState(1234)

	layer_input = self.X
	shape = (self.in_size, self.hidden_size[0])
	embed = EmbeddingLayer("e", shape, layer_input)
	self.layers.append(embed)
	self.params += embed.params

        # hidden layers
        for i in xrange(len(self.hidden_size)):
            if i == 0:
                layer_input = embed.activation
                shape = (self.hidden_size[0], self.hidden_size[0])
            else:
                layer_input = self.layers[i - 1].activation
                shape = (self.hidden_size[i - 1], self.hidden_size[i])

            hidden_layer = LSTMLayer(rng, str(i), shape, layer_input,
            	self.batch_size, 0.0)
            
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        # output layer
        output_layer = SoftmaxLayer((hidden_layer.out_size, self.out_size),
                                    hidden_layer.activation, self.batch_size)
        self.layers.append(output_layer)
        self.params += output_layer.params
   
    # https://github.com/fchollet/keras/pull/9/files
        self.epsilon = 1.0e-15
    def categorical_crossentropy(self, y_pred, y_true):
        y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        ce = T.nnet.categorical_crossentropy(y_pred, y_true)
        return T.mean(ce)

    def define_train_test_funcs(self):
        activation = self.layers[len(self.layers) - 1].activation
        self.Y = T.tensor3("Y")
        cost = self.categorical_crossentropy(activation, self.Y)
        
        gparams = []
        for param in self.params:
            #gparam = T.grad(cost, param)
            gparam = T.clip(T.grad(cost, param), -3, 3)
            gparams.append(gparam)

        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)
        
        self.train = theano.function(inputs = [self.index, lr, self.batch_size],
                                               outputs = cost,
                                               updates = updates,
					       on_unused_input='warn',
					       givens={
						       self.X: self.Xdata[self.index],
						       self.Y: self.Ydata[self.index]
					       })
