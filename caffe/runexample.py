import os
import numpy as np
from timeit import default_timer as timer
import gc
import argparse
import gzip
import six.moves.cPickle as pickle


FILENAME="/home/ec2-user/Knet8-Benchmarks/data/mnist.pkl.gz"


def load_mnist(dataset, flatten=False):
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    train_set = (
        np.concatenate((train_set[0], valid_set[0])),
        np.concatenate((train_set[1], valid_set[1])),
    )
     
    xtrn, ytrn = train_set
    xtst, ytst = test_set

    h, w = 28, 28
    if flatten:
        h, w = 1, 784

    xtrn = xtrn.reshape(xtrn.shape[0], 1, h, w).astype(np.float32)
    ytrn = ytrn.reshape(ytrn.shape[0], 1, 1, 1).astype(np.float32)
    xtst = xtst.reshape(xtst.shape[0], 1, h, w).astype(np.float32)
    ytst = ytst.reshape(ytst.shape[0], 1, 1, 1).astype(np.float32)

    return (xtrn, ytrn, xtst, ytst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, type=str, help="solver.prototxt")
    parser.add_argument("--gpu", action="store_true", help="use gpu")
    parser.add_argument("--memory", action="store_true", help="use memory data layer")
    parser.add_argument("--iter", type=int, help="number of iterations")
    parser.add_argument("--debug", action="store_true", help="enable glog output")
    parser.add_argument("--data", default=FILENAME, type=str, help="mnist pickle file path")
    parser.add_argument("--batchsize", default=100, type=int, help="batch size")
    parser.add_argument("--flatten", action="store_true", help="flatten mnist data")
    args = parser.parse_args()

    print args

    if not args.debug:
        os.environ['GLOG_minloglevel'] = '2'
    import caffe

    data = (0, 0, 0, 0)
    if args.memory:
        data = load_mnist(FILENAME, flatten=args.flatten)

    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()

    solver = caffe.SGDSolver(os.path.abspath(args.solver))
    gc.disable()
    t0 = timer()

    if not args.memory:
        solver.solve()
    else:
        for i in range(args.iter):
            lower, upper = i*args.batchsize, (i+1)*args.batchsize
            solver.net.set_input_arrays(
                data[0][lower:upper,:,:,:], data[1][lower:upper,:,:,:])
            solver.step(1)
    t1 = timer()
    gc.enable()
    
    print "Time: %.4f" % (t1 - t0)
    if args.memory:
        print "Iterations: %d" % (args.iter,)

if __name__ == "__main__":
    main()
