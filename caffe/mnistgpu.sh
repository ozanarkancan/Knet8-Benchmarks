#!/bin/sh

printf "MLP GPU (NOT FLATTEN LMDB)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver mlp_solver.prototxt --gpu
done

printf "MLP GPU (MEMORY DATA PER BATCH FORCE LOAD)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver mlp_solver2.prototxt --gpu --flatten --memory --iter 6000
done

printf "MLP GPU (MEMORY DATA FORCE BULK LOAD)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver mlp_solver2.prototxt --gpu --flatten --memory --bulk
done

printf "SOFTMAX GPU (NOT FLATTEN LMDB)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver softmax_solver.prototxt --gpu
done

printf "SOFTMAX GPU (MEMORY DATA PER BATCH FORCE LOAD)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver softmax_solver2.prototxt --gpu --flatten --memory --iter 6000
done

printf "SOFTMAX GPU (MEMORY DATA FORCE BULK LOAD)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver softmax_solver2.prototxt --gpu --flatten --memory --bulk
done
