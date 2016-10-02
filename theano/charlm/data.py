# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip
import jieba

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

def char_sequence(f_path = None, batch_size = 1):
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    text = open(f_path).read()
    print "Corpus length: {}".format(len(text))
    chars = sorted(list(set(text)))
    w2i = dict((c, i) for i, c in enumerate(chars))
    i2w = dict((i, c) for i, c in enumerate(chars))

    maxlen = 100
    step = 100
    for i in range(0, len(text) - maxlen + 1, step):
        lines.append(text[i: i + maxlen])

    for i in range(0, len(lines)):
        line = lines[i]
        x = np.zeros((len(line), len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(line)):
            x[j, w2i[line[j]]] = 1
        seqs.append(np.asmatrix(x))

    data_xy = batch_sequences(seqs, i2w, w2i, batch_size)
    print "#dic = " + str(len(w2i))
    return seqs, i2w, w2i, data_xy

def batch_sequences(seqs, i2w, w2i, batch_size):
    data_xy = {}
    batch_x = []
    batch_y = []
    XX = []
    YY = []
    b_lengths = []
    seqs_len = []
    batch_id = 0
    dim = len(w2i)
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)
    for i in xrange(len(seqs)):
        seq = seqs[i];
        X = seq[0 : len(seq) - 1, ]
        Y = seq[1 : len(seq), ]
        batch_x.append(X)
        seqs_len.append(X.shape[0])
        batch_y.append(Y)

        if len(batch_x) == batch_size or (i == len(seqs) - 1):
            max_len = np.max(seqs_len);
            concat_X = np.zeros((max_len, len(batch_x), dim), dtype = theano.config.floatX)
            concat_Y = concat_X.copy()
            mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
            for b_i in xrange(len(batch_x)):
                X = batch_x[b_i]
                Y = batch_y[b_i]
                mask[0 : X.shape[0], b_i] = 1
                for r in xrange(max_len - X.shape[0]):
                    X = np.concatenate((X, zeros_m), axis=0)
                    Y = np.concatenate((Y, zeros_m), axis=0)
                concat_X[:, b_i, :] = X 
                concat_Y[:, b_i, :] = Y
	    if batch_id == 16:
		    continue
	    XX.append(concat_X)
	    YY.append(concat_Y)
	    b_lengths.append(len(batch_x))
            #data_xy[batch_id] = [concat_X, concat_Y, mask, mask, len(batch_x)]
            batch_x = []
            batch_y = []
            seqs_len = []
            batch_id += 1
    return (XX, YY, b_lengths)

# limit memory
def batch_index(seqs, i2w, w2i, batch_size):
    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    for i in xrange(len(seqs)):
        batch_x.append(i)
        batch_y.append(i)
        if len(batch_x) == batch_size or (i == len(seqs) - 1):
            data_xy[batch_id] = [batch_x, batch_y, [], len(batch_x)]
            batch_x = []
            batch_y = []
            batch_id += 1
    return data_xy

def index2seqs(lines, x_index, w2i):
    seqs = []
    for i in x_index:
        line = lines[i]
        x = np.zeros((len(line), len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(line)):
            x[j, w2i[line[j]]] = 1
        seqs.append(np.asmatrix(x))

    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    dim = len(w2i)
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)
    for i in xrange(len(seqs)):
        seq = seqs[i];
        X = seq[0 : len(seq) - 1, ]
        Y = seq[1 : len(seq), ]
        batch_x.append(X)
        seqs_len.append(X.shape[0])
        batch_y.append(Y)

    max_len = np.max(seqs_len);
    mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
    concat_X = np.zeros((max_len, len(batch_x), dim), dtype = theano.config.floatX)
    concat_Y = concat_X.copy()
    
    for b_i in xrange(len(batch_x)):
        X = batch_x[b_i]
        Y = batch_y[b_i]
        mask[0 : X.shape[0], b_i] = 1
        for r in xrange(max_len - X.shape[0]):
            X = np.concatenate((X, zeros_m), axis=0)
            Y = np.concatenate((Y, zeros_m), axis=0)
        concat_X[:, b_i, :] = X 
        concat_Y[:, b_i, :] = Y
    return concat_X, concat_Y, mask, len(batch_x)
