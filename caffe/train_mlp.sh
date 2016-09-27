#!/usr/bin/env sh
set -e

caffe.bin train --solver=mlp_solver.prototxt $@
