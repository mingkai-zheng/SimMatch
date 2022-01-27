#!/bin/bash


job_name=$1
train_gpu=$2
num_node=$3
command=$4
total_process=$((train_gpu*num_node))

mkdir -p log


port=$(( $RANDOM % 300 + 23450 ))


# nohup
GLOG_vmodule=MemcachedClient=-1 \
srun --partition=pat_op \
--mpi=pmi2 -n$total_process \
--gres=gpu:$train_gpu \
--ntasks-per-node=$train_gpu \
--job-name=$job_name \
--kill-on-bad-exit=1 \
--cpus-per-task=14 \
$command --port $port 2>&1|tee -a log/$job_name.log &

