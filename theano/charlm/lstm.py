#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class LSTMLayer(object):
    def __init__(self, rng, layer_id, shape, X, batch_size = 1, p = 0.5):
        prefix = "LSTM_"
        layer_id = "_" + layer_id
        self.in_size, self.out_size = shape
        
        self.W_x_ifoc = init_weights_4((self.in_size, self.out_size), prefix + "W_x_ifoc" + layer_id)
        self.W_h_ifoc = init_weights_4((self.out_size, self.out_size), prefix + "W_h_ifoc" + layer_id)
        self.W_c_o = init_weights((self.out_size, self.out_size), prefix + "W_c_o" + layer_id)
        self.b_ifoc = init_bias(self.out_size * 4, prefix + "b_ifoc" + layer_id)

        self.params = [self.W_x_ifoc, self.W_h_ifoc, self.b_ifoc]

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim : (n + 1) * dim]
            return _x[:, n * dim : (n + 1) * dim]

        X_4ifoc = T.dot(X, self.W_x_ifoc) + self.b_ifoc
        def _active(x_4ifoc, pre_h, pre_c, W_h_ifoc):
            ifoc_preact = x_4ifoc + T.dot(pre_h, W_h_ifoc)

            i = T.nnet.sigmoid(_slice(ifoc_preact, 0, self.out_size))
            f = T.nnet.sigmoid(_slice(ifoc_preact, 1, self.out_size))
            gc = T.tanh(_slice(ifoc_preact, 2, self.out_size))
            c = f * pre_c + i * gc
            o = T.nnet.sigmoid(_slice(ifoc_preact, 3, self.out_size))
            h = o * T.tanh(c)

            return h, c
        [h, c], updates = theano.scan(_active,
                                      sequences = [X_4ifoc],
                                      outputs_info = [T.alloc(floatX(0.), batch_size, self.out_size),
                                                      T.alloc(floatX(0.), batch_size, self.out_size)],
                                      non_sequences = [self.W_h_ifoc],
                                      strict = True)
        self.activation = h
