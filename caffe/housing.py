import os
#os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import six.moves.cPickle as pickle
import gzip
from timeit import default_timer as timer
import gc
import argparse
import os


FILE_PATH = "/home/ec2-user/Knet8-Benchmarks/data/housing.data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="housing_solver.prototxt", type=str, help="solver.prototxt")
    parser.add_argument("--gpu", action="store_true", help="use gpu")
    parser.add_argument("--debug", action="store_true", help="enable glog output")
    parser.add_argument("--data", type=str, default=FILE_PATH, help="houing.data file")
    args = parser.parse_args()
    print args

    data = np.loadtxt(os.path.abspath(args.data), dtype=np.float32)
    x = data[:, 0:13]
    x = (x - x.mean(0)) / x.std(0)
    y = data[:, 13:]
    xtrn = np.zeros((506, 1, 1, 13))
    xtrn[:,0,0,:] = x
    ytrn = y.reshape((506, 1, 1, 1))

    if not args.debug:
        os.environ['GLOG_minloglevel'] = '2'
    import caffe

    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()

    solver = caffe.SGDSolver(os.path.abspath(args.solver))
    gc.disable()
    t0 = timer()
    for i in range(10000):
        solver.net.set_input_arrays(xtrn.astype(np.float32), ytrn.astype(np.float32))
        solver.step(1)
    t1 = timer()

    gc.enable()
    print "Time: %.4f" % (t1 - t0)


if __name__ == "__main__":
    main()
