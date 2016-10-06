#!/usr/bin/env python

import numpy as np
import h5py
import os


SOURCE="/home/ec2-user/Knet8-Benchmarks/data/19.txt"
with open(SOURCE, 'r') as f:
    txt = [c for c in f.read()] 

timesteps = 100
batchsize = 128
i, vocab = 0, dict()
for c in set(txt):
    vocab[c] = i
    i += 1

# clip last batch
txt = txt[:-(len(txt) % (batchsize*timesteps))]

# source, target = [], []
source = np.zeros((99*16,128), dtype=np.uint8)
target = np.zeros((99*16,128), dtype=np.uint8)
cont = np.ones(source.shape).astype(np.uint8)
for i in range(cont.shape[0]):
    if i % 99 == 0:
        cont[i] = 0

i, j, k = 0, 0, 0
while True:
    #print i, j, k
    lo, up = i * timesteps, (i+1) * timesteps
    so = np.array(map(lambda x: vocab[x], txt[lo:up-1])).astype(np.uint8)
    ta = np.array(map(lambda x: vocab[x], txt[lo+1:up])).astype(np.uint8)
    source[j:j+timesteps-1,k] = so.reshape(1,99)
    target[j:j+timesteps-1,k] = ta.reshape(1,99)
    k += 1
    if k >= 128:
        k = 0
        j = j + timesteps - 1
    i += 1
    if j >= source.shape[0]:
        break


f = h5py.File("data.h5", "w")
f.create_dataset('data', data=source)
f.create_dataset('label', data=target)
f.create_dataset('cont', data=cont)
f.close()

print 'Done.'
print 'vocab; %d' % (len(vocab))
