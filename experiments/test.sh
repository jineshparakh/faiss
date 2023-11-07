#!/bin/bash

make -C build composite_ivf_hnsw

use_preexisting=0
index_factory_string="IVF165_HNSW128,Flat"
training_set="sift/sift_learn.fvecs"
base_set="sift/sift_base.fvecs"
base_train_same=0
query_set="sift/sift_query.fvecs"
groundtruth_set="sift/sift_groundtruth.ivecs"
index_params="nprobe=5"
index_storage_directory="experiments/indexes/$index_factory_string-$base_train_same.index"
logs_file="experiments/logs/$index_factory_string-$use_preexisting$base_train_same-$index_params.log"



# ./build/experiments/composite_ivf_hnsw use_preexisting index learn_set base_set index_store_location query_set groundtruth_set
# ./build/experiments/composite_ivf_hnsw 0 IVF10_HNSW,Flat sift/sift_learn.fvecs sift/sift_base.fvecs experiments/IVF10_HNSW,Flat sift/sift_query.fvecs sift/sift_groundtruth.ivecs

./build/experiments/composite_ivf_hnsw $use_preexisting $index_factory_string $training_set $base_set $index_storage_directory $query_set $groundtruth_set $index_params &> $logs_file
# ./build/experiments/composite_ivf_hnsw 1 nil nil nil experiments/index.index sift/sift_query.fvecs sift/sift_groundtruth.ivecs