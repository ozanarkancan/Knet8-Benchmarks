import os
import gzip
import numpy as np
import lmdb
import caffe
import six.moves.cPickle as pickle

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

    xtrn = xtrn.reshape(xtrn.shape[0], 1, h, w).astype(np.float32)
    ytrn = ytrn.reshape(ytrn.shape[0], 1, 1, 1).astype(np.float32)
    xtst = xtst.reshape(xtst.shape[0], 1, h, w).astype(np.float32)
    ytst = ytst.reshape(ytst.shape[0], 1, 1, 1).astype(np.float32)

    # LMDB part for trn
    savedir = os.path.abspath(os.path.join(target, "trn"))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    trndb = lmdb.open(savedir, map_size=int(1e12), map_async=True, writemap=True)
    trntxn = trndb.begin(write=True)

    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = 1
    if flatten:
        datum.height = 1
        datum.width = 784
    else:
        datum.height = 28
        datum.width = 28
    
    for i in range(xtrn.shape[0]):
        x = xtrn[i]
        y = ytrn[i]
        datum.data = x.tobytes()
        datum.label = int(y)
        trntxn.put('{:0>10d}'.format(i), datum.SerializeToString())
    trntxn.commit()
    trndb.close()

    # LMDB part for tst
    savedir = os.path.abspath(os.path.join(target, "tst"))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    tstdb = lmdb.open(savedir, map_size=int(1e12), map_async=True, writemap=True)
    tsttxn = tstdb.begin(write=True)

    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = 1
    if flatten:
        datum.height = 1
        datum.width = 784
    else:
        datum.height = 28
        datum.width = 28
    
    for i in range(xtst.shape[0]):
        x = xtst[i]
        y = ytst[i]
        datum.data = x.tobytes()
        datum.label = int(y)
        tsttxn.put('{:0>10d}'.format(i), datum.SerializeToString())
    tsttxn.commit()
    tstdb.close()


def main():
    make_mnist_lmdb(SOURCE, "caffe_mnist", flatten=True) 
    print "Done"

if __name__ == "__main__":
    main()
