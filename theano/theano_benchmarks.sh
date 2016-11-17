#!/bin/bash

export PATH="/home/ec2-user/anaconda2/bin:$PATH"
device=$1

printf "Theano Benchmarks Using: $device \n"

printf "*** Housing Example ***"

for i in `seq 1 6`; do
	THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=1,allow_gc=False python housing_theano.py
done

printf "\n*** MNIST Softmax Example ***"

for i in `seq 1 6`; do
	THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=1,allow_gc=False python mnist_softmax_theano.py
done

printf "\n*** MNIST Mlp Example ***"

for i in `seq 1 6`; do
	THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=1,allow_gc=False python mnist_mlp_theano.py
done

printf "\n*** MNIST Lenet Example ***"

for i in `seq 1 6`; do
	THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=1,allow_gc=False python mnist_lenet_theano.py
done

cd charlm

printf "\n*** Charlm Example ***"

for i in `seq 1 6`; do
	THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=1,allow_gc=False python main_char_train.py
done

