#!/bin/bash

printf "CPU BULK RUN\n\n"

printf "HOUSING\n"
for i in `seq 1 5`; do
	python linreg.py 
done

printf "SOFTMAX\n"
for i in `seq 1 5`; do
	python tensorflow_softmax.py
done

printf "MLP\n"
for i in `seq 1 5`; do
	python tensorflow_mlp2.py
done

printf "LENET\n"
for i in `seq 1 5`; do
	python tensorflow_lenet.py
done

printf "CHARLM\n"
for i in `seq 1 5`; do
	python train.py
done 
