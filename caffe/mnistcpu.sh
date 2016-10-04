#!/bin/sh

printf "LENET CPU (LMDB)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver lenet_solver.prototxt
done

printf "MLP CPU (NOT FLATTEN LMDB)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver mlp_solver.prototxt
done

printf "MLP CPU (MEMORY DATA PER BATCH FORCE LOAD)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver mlp_solver2.prototxt --flatten --memory --iter 6000
done

printf "MLP CPU (MEMORY DATA FORCE BULK LOAD)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver mlp_solver2.prototxt --flatten --memory --bulk
done

printf "SOFTMAX CPU (NOT FLATTEN LMDB)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver softmax_solver.prototxt
done

printf "SOFTMAX CPU (MEMORY DATA PER BATCH FORCE LOAD)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver softmax_solver2.prototxt --flatten --memory --iter 6000
done

printf "SOFTMAX CPU (MEMORY DATA FORCE BULK LOAD)\n"
for i in `seq 1 5`; do
	python runmnist.py --solver softmax_solver2.prototxt --flatten --memory --bulk
done
