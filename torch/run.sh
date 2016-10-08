#!/bin/bash

MODE=${1:-"GPUv1"}
RUNTIME=1

printf "Torch Benchmarks on Housing, Softmax, MLP, Lenet and Charlm examples. Selected mode = $MODE .All Possible modes = [GPUv1 GPUv2 CPU]\n"

cd $MODE

printf "*** Housing Example ***\n"
cd housing
for i in `seq 1 $RUNTIME`; do
    th housing.lua
done
cd ..

printf "*** Softmax Example ***\n"
for i in `seq 1 $RUNTIME`; do
    th softmax.lua
done

printf "*** MLP Example ***\n"
for i in `seq 1 $RUNTIME`; do
    th mlp.lua
done

printf "*** Lenet Example ***\n"
for i in `seq 1 $RUNTIME`; do
    th lenet.lua
done

printf "*** Charlm Example ***\n"
cd charlm
# Preprocessing data
python scripts/preprocess.py --input_txt data/19.txt --output_h5 data/my_data.h5 --output_json data/my_data.json
for i in `seq 1 $RUNTIME`; do
    th train.lua -input_h5 data/my_data.h5 -input_json data/my_data.json  -num_layers 1 -rnn_size 256 -speed_benchmark 1
done
cd ..

