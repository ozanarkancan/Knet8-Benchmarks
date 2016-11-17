import os
import gzip
import numpy as np
import lmdb
import caffe
import six.moves.cPickle as pickle
import argparse

SOURCE="/home/ec2-user/Knet8-Benchmarks/data/mnist.pkl.gz"


def make_mnist_lmdb(source, target, flatten=False):
    with gzip.open(source, 'rb') as f:
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

    xtrn = xtrn.reshape(xtrn.shape[0], 1, h, w).astype(np.float)
    ytrn = ytrn.reshape(ytrn.shape[0], 1, 1, 1).astype(np.float)
    xtst = xtst.reshape(xtst.shape[0], 1, h, w).astype(np.float)
    ytst = ytst.reshape(ytst.shape[0], 1, 1, 1).astype(np.float)

    # LMDB part for trn
    savedir = os.path.abspath(os.path.join(target, "trn"))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    trndb = lmdb.open(savedir, map_size=int(1e12), map_async=True, max_dbs=0)
    trntxn = trndb.begin(write=True)

    for i in range(xtrn.shape[0]):
        datum = caffe.io.array_to_datum(xtrn[i], int(ytrn[i,0,0,0]))
        trntxn.put('{:0>10d}'.format(i), datum.SerializeToString())
    trntxn.commit()
    trndb.close()

    # LMDB part for tst
    savedir = os.path.abspath(os.path.join(target, "tst"))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    tstdb = lmdb.open(savedir, map_size=int(1e12), map_async=True, max_dbs=0)
    tsttxn = tstdb.begin(write=True)

    for i in range(xtst.shape[0]):
        datum = caffe.io.array_to_datum(xtst[i], int(ytst[i,0,0,0]))
        tsttxn.put('{:0>10d}'.format(i), datum.SerializeToString())
    tsttxn.commit()
    tstdb.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=str, help="raw MNIST pickle file")
    parser.add_argument("--target", required=True, type=str, help="output directory")
    parser.add_argument("--flatten", action="store_true", help="data has one dimension or not")
    args  = parser.parse_args()

    print args
    make_mnist_lmdb(args.source, args.target, flatten=args.flatten) 
    print "Done"

if __name__ == "__main__":
    main()
