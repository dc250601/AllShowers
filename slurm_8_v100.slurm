#!/usr/bin/env bash
# Pass an account on the sbatch command line if required by your site.
# Override --gres if your cluster uses a different V100 resource name.
#SBATCH --job-name=allshowers-8-v100
#SBATCH --partition=maxgpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=allshowers-%j.out
#SBATCH --error=allshowers-%j.err

set -euo pipefail
export PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
cd "$PROJECT_DIR"
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export PYTHON_BIN=${PYTHON_BIN:-python}
export USE_WANDB=${USE_WANDB:-False}

if [[ -n ${ENV_SETUP:-} ]]; then
    source "$ENV_SETUP"
else
    conda_bin=${CONDA_EXE:-conda}
    eval "$("$conda_bin" shell.bash hook)"
    conda activate "${CONDA_ENV_PATH:-$HOME/AllShowers}"
fi

config=${1:-conf/transformer.yaml}
if (( $# > 0 )); then shift; fi
export RESULT_PATH=${RESULT_PATH:-$PROJECT_DIR/results/slurm-$SLURM_JOB_ID}
mkdir -p "$RESULT_PATH"
RESULT_PATH=$(cd "$RESULT_PATH" && pwd -P)
export RESULT_PATH
launch_config="$RESULT_PATH/launch-$SLURM_JOB_ID.yaml"

# Default keeps the previous 64 samples/GPU: 8 nodes * 4 GPUs * 64 = 2048.
# GLOBAL_BATCH_SIZE=256 preserves the old global batch (8 samples/GPU).
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE * 64 ))}
"$PYTHON_BIN" - "$config" "$launch_config" <<'PY'
import os
import sys
import yaml
with open(sys.argv[1]) as source:
    conf = yaml.safe_load(source)
batch = int(os.environ["GLOBAL_BATCH_SIZE"])
world = int(os.environ["SLURM_JOB_NUM_NODES"]) * int(os.environ["GPUS_PER_NODE"])
if batch < world or batch % world:
    raise SystemExit(f"GLOBAL_BATCH_SIZE must be a positive multiple of {world}.")
conf["train"]["batch_size"] = batch
conf["result_path"] = os.environ["RESULT_PATH"]
with open(sys.argv[2], "w") as target:
    yaml.safe_dump(conf, target, sort_keys=False)
print(f"Global batch={batch}; batch/GPU={batch // world}; results={conf['result_path']}")
PY

mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=${MASTER_ADDR:-${hosts[0]}}
export MASTER_PORT=${MASTER_PORT:-$(( 20000 + SLURM_JOB_ID % 20000 ))}
echo "Starting $SLURM_JOB_NUM_NODES nodes, $GPUS_PER_NODE GPUs/node; config=$launch_config"

# One launcher per node; each launcher must see all four allocated GPUs.
srun --nodes="$SLURM_JOB_NUM_NODES" \
    --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpu-bind=none --kill-on-bad-exit=1 --export=ALL \
    bash "$PROJECT_DIR/script_muti_node.sh" "$launch_config" "$RESULT_PATH" "$@"
