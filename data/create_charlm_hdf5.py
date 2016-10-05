#!/usr/bin/env python

import numpy as np
import h5py
import os


SOURCE="/home/ec2-user/Knet8-Benchmarks/data/19.txt"
with open(SOURCE, 'r') as f:
    txt = [c for c in f.read()] 
   
i, vocab = 0, dict()
for c in set(txt):
    vocab[c] = i
    i += 1

timesteps = 100
batchsize = 128
nsamples = len(txt) / timesteps
source, target = [], []

for i in range(0, len(txt), timesteps)[:-1]:
    source.append([vocab[txt[j]] for j in range(i, i+timesteps-1)])
    target.append([vocab[txt[j]] for j in range(i+1, i+timesteps)])


source = np.array(source).astype(np.uint8)
target = np.array(target).astype(np.uint8)
cont = np.ones(source.shape).astype(np.uint8)
cont[:,0] = 0

f = h5py.File("data.h5", "w")
f.create_dataset('data', data=source)
f.create_dataset('label', data=target)
f.create_dataset('cont', data=cont)
f.close()

print 'Done.'
print 'vocab; %d, nbatches: %d' % (len(vocab), nsamples/batchsize)
