#!/bin/bash

printf "GPU BULK RUN\n\n"

printf "HOUSING\n"
for i in `seq 1 5`; do
	python housing.py --gpu
done

printf "SOFTMAX\n"
for i in `seq 1 5`; do
	python runmnist.py --solver softmax_solver.prototxt --gpu
done

printf "MLP\n"
for i in `seq 1 5`; do
	python runmnist.py --solver mlp_solver.prototxt --gpu
done

printf "LENET\n"
for i in `seq 1 5`; do
	python runmnist.py --solver lenet_solver.prototxt --gpu
done

printf "CHARLM\n"
for i in `seq 1 5`; do
	python runcharlm.py --solver charlm_solver.prototxt --gpu
done 
