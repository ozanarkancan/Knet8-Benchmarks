#!/bin/bash

printf "CPU BULK RUN\n\n"

printf "HOUSING\n"
for i in `seq 1 5`; do
	python housing.py 
done

printf "SOFTMAX\n"
for i in `seq 1 5`; do
	python runmnist.py --solver softmax_solver.prototxt 
done

printf "MLP\n"
for i in `seq 1 5`; do
	python runmnist.py --solver mlp_solver.prototxt 
done

printf "LENET\n"
for i in `seq 1 5`; do
	python runmnist.py --solver lenet_solver.prototxt 
done

printf "CHARLM\n"
for i in `seq 1 5`; do
	python runcharlm.py --solver charlm_solver.prototxt 
done 
