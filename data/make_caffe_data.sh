#!/bin/bash

python create_charlm_hdf5.py
python create_mnist_lmdb.py --source mnist.pkl.gz --target caffe_mnist_flatten
python create_mnist_lmdb.py --source mnist.pkl.gz --target caffe_mnist_flatten --flatten
