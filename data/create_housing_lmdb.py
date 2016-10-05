import os
import numpy as np
import lmdb
import caffe
import six.moves.cPickle as pickle


def make_lmdb(data, savedir):
    savedir = os.path.abspath(os.path.join(savedir))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    xdb = lmdb.open(savedir, map_size=int(1e12), map_async=True, writemap=True)
    xtxn = xdb.begin(write=True)

    data = data.transpose()
    n, m = 13, 506
    x = data[0:n]
    y = data[13,:]
    x = (x - x.mean(1).reshape(-1,1)) / x.std(1).reshape(-1,1)


    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = 1
    datum.height = 1
    datum.width = 13
    for i in range(m):
        datum.data = x[:,i].reshape(13,1).tobytes()
        datum.label = int(y[i])
        xtxn.put('{:0>10d}'.format(i), datum.SerializeToString())
    xtxn.commit()
    xdb.close()


def main():
    data = np.loadtxt("housing.data", dtype=np.float32)
    make_lmdb(data, "caffe-housing") 
    print "Done"


if __name__ == "__main__":
    main()
