#!/usr/bin/env sh
set -e

caffe.bin train --solver=lenet_solver.prototxt $@
