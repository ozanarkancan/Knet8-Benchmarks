#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *
import data

use_gpu(0)

e = 0.01
lr = 4.0
drop_rate = 0.
batch_size = 128
hidden_size = [256]
# try: gru, lstm
cell = "lstm"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "sgd" 

seqs, i2w, w2i, data_xy = data.char_sequence("../../data/19.txt", batch_size)
dim_x = len(w2i)
dim_y = len(w2i)
print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = RNN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate)

print "training..."
start = time.time()
g_error = 9999.9999
count = 0
for i in xrange(1):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_xy.items():
	count += 1
        X = xy[0] 
        Y = xy[1]
        maskX = xy[2]
        maskY = xy[3]
        local_batch_size = xy[4]
        cost = model.train(X, maskX, Y, maskY, lr, local_batch_size)
        error += cost
    in_time = time.time() - in_start

    error /= count;

    print "Iter = " + str(i) + ", Loss = " + str(error) + ", Time = " + str(in_time)
