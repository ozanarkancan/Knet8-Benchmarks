import os
import gzip
import numpy as np
import lmdb
import caffe
import six.moves.cPickle as pickle


def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    train_set = (
        np.concatenate((train_set[0], valid_set[0])),
        np.concatenate((train_set[1], valid_set[1])),
    )

    return (train_set, test_set)


def make_lmdb(data, filename):
    N = data[0].shape[0]
    X = np.zeros((N, 1, 1, 784), dtype=np.float32)
    X[:,0,0,:] = data[0]
    y = data[1]
    map_size = X.nbytes * 10
    env = lmdb.open(filename, map_size=map_size)

    with env.begin(write=True) as txn:
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i,:,:,:].tobytes()
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


if __name__ == "__main__":
    trn, tst = load_data("mnist.pkl.gz")
    make_lmdb(trn, "mnist1d_trn_lmdb")
    make_lmdb(tst, "mnist1d_tst_lmdb")
