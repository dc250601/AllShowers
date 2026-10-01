#!/usr/bin/env bash
# Run once per node, normally via slurm_8_v100.slurm.
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: bash script_muti_node.sh CONFIG RESULT_PATH [training arguments...]" >&2
    exit 2
fi
config=$1
result_path=$2
shift 2

: "${MASTER_ADDR:?Set MASTER_ADDR to the first allocated node}"
: "${MASTER_PORT:?Set MASTER_PORT to a free port on that node}"
nnodes=${SLURM_JOB_NUM_NODES:-${NNODES:-8}}
node_rank=${SLURM_PROCID:-${NODE_RANK:?Set NODE_RANK outside Slurm}}
gpus_per_node=${GPUS_PER_NODE:-4}
python_bin=${PYTHON_BIN:-python}

# ENV_SETUP may load modules and activate conda/venv; it must exist on every node.
if [[ -n ${ENV_SETUP:-} ]]; then
    source "$ENV_SETUP"
else
    conda_bin=${CONDA_EXE:-conda}
    eval "$("$conda_bin" shell.bash hook)"
    conda activate "${CONDA_ENV_PATH:-$HOME/AllShowers}"
fi

cpus=${SLURM_CPUS_PER_TASK:-72}
threads=$(( cpus / gpus_per_node ))
(( threads > 0 )) || threads=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$threads}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

"$python_bin" - "$gpus_per_node" <<'PY'
import sys
import torch
expected = int(sys.argv[1])
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(f"Expected {expected} visible GPUs; found {actual}. Check Slurm GPU allocation.")
PY

echo "host=$(hostname) node_rank=$node_rank nodes=$nnodes GPUs/node=$gpus_per_node threads/worker=$OMP_NUM_THREADS master=$MASTER_ADDR:$MASTER_PORT"
exec "$python_bin" -m torch.distributed.run \
    --nnodes="$nnodes" \
    --nproc-per-node="$gpus_per_node" \
    --node-rank="$node_rank" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --max-restarts=0 \
    --module allshowers.train \
    "$config" --ddp --result_path "$result_path" \
    --wandb "${USE_WANDB:-False}" "$@"
