import numpy as np
from timeit import default_timer as timer
import gc
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, type=str, help="solver.prototxt")
    parser.add_argument("--gpu", action="store_true", help="use gpu")
    parser.add_argument("--debug", action="store_true", help="enable glog output")
    args = parser.parse_args()

    import os
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
    solver.solve()
    t1 = timer()
    gc.enable()
    
    print "Time: %.4f" % (t1 - t0)


if __name__ == "__main__":
    main()
