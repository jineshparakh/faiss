#!/bin/bash
# export OMP_WAIT_POLICY=PASSIVE 
make -C build ivf_hnsw

use_preexisting=1
learn_vectors=1000000
centroids_percent=0.01
nprobe_percent=0.1

M=128
nlist=$(echo $centroids_percent $learn_vectors | awk '{printf("%d\n", $1*$2)}')
echo $nlist
training_set="sift/sift_base.fvecs"
base_set="sift/sift_base.fvecs"
base_train_same=1
query_set="sift/sift_query.fvecs"
groundtruth_set="sift/sift_groundtruth.ivecs"
num_nprobe=$(echo $nprobe_percent $nlist | awk '{printf("%d\n", $1*$2)}')
echo $num_nprobe
train_params="efConstruction=120"
hnsw_search_params="efSearch=512"
search_params="nprobe=$num_nprobe"
index_params=$train_params","$search_params","$hnsw_search_params
echo $index_params
omp_set_num_threads=28
index_storage_directory="experiments/indexes/ivf_hnsw/1%/$nlist-$M-$train_params-$base_train_same.index"
logs_file="experiments/logs/ivf_hnsw/1%/$train_params/$hnsw_search_params/$nlist-$M-$use_preexisting$base_train_same-$index_params-$omp_set_num_threads.log"


./build/experiments/ivf_hnsw $use_preexisting $M $nlist $training_set $base_set $index_storage_directory $query_set $groundtruth_set $index_params $omp_set_num_threads &> $logs_file
