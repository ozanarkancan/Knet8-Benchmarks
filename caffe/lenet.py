import os
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import caffe
import six.moves.cPickle as pickle
import gzip
from timeit import default_timer as timer
import gc


file_path = "/home/ec2-user/Knet8-Benchmarks/data/mnist.pkl.gz"
with gzip.open(file_path, 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

train_set = (
    np.concatenate((train_set[0], valid_set[0])),
    np.concatenate((train_set[1], valid_set[1])),
)

xtrn, ytrn = train_set
xtst, ytst = test_set

xtrn = xtrn.reshape(xtrn.shape[0], 1, 28, 28)
ytrn = ytrn.reshape(ytrn.shape[0], 1, 1, 1).astype(np.float32)
xtst = xtst.reshape(xtst.shape[0], 1, 1, xtst.shape[1])
ytst = ytst.reshape(ytst.shape[0], 1)
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('lenet_solver.prototxt')
solver.net.set_input_arrays(xtrn, ytrn)
gc.disable()
t0 = timer()
solver.solve()
t1 = timer()
gc.enable()
print "Time: %.4f" % (t1 - t0)
