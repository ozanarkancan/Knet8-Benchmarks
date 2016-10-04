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
x = (x - x.mean(0)) / x.std(0)
y = data[:, 13:]
xtrn = np.zeros((506, 1, 1, 13))
xtrn[:,0,0,:] = x
ytrn = y.reshape((506, 1, 1, 1))
#caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net("housing.prototxt", caffe.TRAIN)
solver = caffe.SGDSolver('housing_solver.prototxt')
gc.disable()
t0 = timer()
for i in range(10000):
    solver.net.set_input_arrays(xtrn.astype(np.float32), ytrn.astype(np.float32))
    solver.step(1)
t1 = timer()

gc.enable()
print "Time: %.4f" % (t1 - t0)
