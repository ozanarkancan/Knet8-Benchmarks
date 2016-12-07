#!/bin/sh
julia benchmark.jl housingcpu
julia benchmark.jl softmaxcpu
julia benchmark.jl mlpcpu
#julia benchmark.jl lenet
julia benchmark.jl charlmcpu
