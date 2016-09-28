import os
#os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import caffe
import six.moves.cPickle as pickle
import gzip
from timeit import default_timer as timer
import gc


file_path = "/home/ec2-user/Knet8-Benchmarks/data/housing.data"

data = np.loadtxt(file_path, dtype=np.float32)
x = data[:, 0:13]
y = data[:, 13:]

xtrn = np.zeros((506, 1, 1, 13))
ytrn = np.zeros((506, 1, 1, 1))

xtrn[:,0,0,:] = x
ytrn[:,0,0,:] = y

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('housing_solver.prototxt')
solver.net.set_input_arrays(xtrn.astype(np.float32), ytrn.astype(np.float32))
gc.disable()
t0 = timer()
solver.solve()
t1 = timer()
gc.enable()
print "Time: %.4f" % (t1 - t0)
