#!/usr/bin/env sh
set -e

log_path="logs/"
mkdir -p $log_path

save_model_path="caffemodel/"
mkdir -p $save_model_path

# training log
file_prefix="mnist_"
log_file=$(date -d "today" +"%Y-%m-%d-%H:%M:%S")
log_file=$log_path$file_prefix$log_file".log"

# caffe execute file path
caffe_bin="../../../build/tools/caffe"

$caffe_bin train --solver=lenet_solver.prototxt  2>&1  | tee -a $log_file
