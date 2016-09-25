#!/usr/bin/env sh
set -e

caffe.bin train --solver=softmax_solver.prototxt $@
