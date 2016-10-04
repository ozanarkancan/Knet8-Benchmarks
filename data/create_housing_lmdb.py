import os
import numpy as np
import lmdb
import caffe
import six.moves.cPickle as pickle


def chunks(l, n):
    return [l[i:i+n] for i in xrange(0, len(l), n)]


def make_lmdb(data, savedir):
    savedir = os.path.abspath(savedir) 
    xdir = os.path.join(savedir, "xtrn")
    ydir = os.path.join(savedir, "ytrn")
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    xdb = lmdb.open(xdir, map_size=int(1e12), map_async=True, writemap=True)
    ydb = lmdb.open(ydir, map_size=int(1e12), map_async=True, writemap=True)

    xtxn = xdb.begin(write=True)
    ytxn = ydb.begin(write=True)

    data = data.transpose()
    n, m = data.shape[0] - 1, data.shape[1]
    print n, m
    
    for i in range(m):
        # import pdb;pdb.set_trace()
        x = caffe.io.array_to_datum(data[0:n,i].reshape(1,1,13).astype(np.float))
        y = caffe.io.array_to_datum(data[n,i].reshape(1,1,1).astype(np.float))
        xtxn.put('{:0>10d}'.format(i), x.SerializeToString())
        ytxn.put('{:0>10d}'.format(i), y.SerializeToString())
    xtxn.commit()
    ytxn.commit()
    xdb.close()
    ydb.close()


def main():
    data = np.loadtxt("housing.data", dtype=np.float32)
    make_lmdb(data, "caffe-housing") 
    print "Done"


if __name__ == "__main__":
    main()
