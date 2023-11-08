#!/bin/bash

make -C build ivf_hnsw

use_preexisting=0
M=128
nlist=166
training_set="sift/sift_learn.fvecs"
base_set="sift/sift_base.fvecs"
base_train_same=0
query_set="sift/sift_query.fvecs"
groundtruth_set="sift/sift_groundtruth.ivecs"
index_params="nprobe=5"
index_storage_directory="experiments/indexes/ivf_hnsw$nlist-$M-$base_train_same.index"
logs_file="experiments/logs/ivf_hnsw/$nlist-$M-$use_preexisting$base_train_same-$index_params.log"


./build/experiments/ivf_hnsw $use_preexisting $M $nlist $training_set $base_set $index_storage_directory $query_set $groundtruth_set $index_params &> $logs_file
