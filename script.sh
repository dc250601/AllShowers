#!/bin/env bash

num_cpus=$(nproc --all)
num_gpus=$(nvidia-smi -L | wc -l)

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29400
export OMP_NUM_THREADS=$((num_cpus / num_gpus))

echo "node: $(hostname)"
echo "number of CPUs: $num_cpus"
echo "number of GPUs: $num_gpus"
grep MemTotal /proc/meminfo

echo "number of threads: $OMP_NUM_THREADS"
echo "master address: $MASTER_ADDR"
echo "master port: $MASTER_PORT"

echo "start time: $(date)"
echo ""

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --module allshowers.train \
    --ddp /home/diptarko/ShowerDistill/AllShowers/conf/transformer.yaml \
    --result_path /home/diptarko/ShowerDistill/AllShowers/results_new_dataloader/