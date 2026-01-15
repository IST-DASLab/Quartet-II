#!/bin/bash

LOG_DIR="/capstor/store/cscs/swissai/a140/codebases/backprop-scaling-laws/logs/slurm/quartetv2_sweep"
mkdir -p "$LOG_DIR"


# >>>>>>>>>
# 1. Model Configs: "MODEL_SIZE_PREFIX:N_LAYER:N_EMBD:N_HEAD:LR:BASE_TOKENS"
MODEL_CONFIGS=(
    "30M:6:640:5:0.0012:3000000000"
    "50M:7:768:6:0.0012:5000000000"
    "100M:8:1024:8:0.0009:10000000000"
    # "200M:10:1280:10:0.00072:20000000000"
)

# 2. Token Multipliers
MULTIPLIERS=(
    0.25
    0.5
    1
    2
    4
    8
)

# 3. Quantization Setups: "GROUP_DIM:SCALE_DTYPE:UNBIASED"
QUANT_SETUPS=(
    # "128:false:false:false"
    "32:false:false:false"
    # "128:true:false:false"
    # "32:true:false:false"
)
# <<<<<<<<<

export STR_MODEL_CONFIGS="${MODEL_CONFIGS[*]}"
export STR_MULTIPLIERS="${MULTIPLIERS[*]}"
export STR_QUANT_SETUPS="${QUANT_SETUPS[*]}"

# Get array sizes
N_MODELS=${#MODEL_CONFIGS[@]}
N_MULTS=${#MULTIPLIERS[@]}
N_QUANTS=${#QUANT_SETUPS[@]}
TOTAL_JOBS=$((N_MODELS * N_MULTS * N_QUANTS))
ARRAY_LIMIT=$((TOTAL_JOBS - 1))

echo "Submitting Job Array with ${TOTAL_JOBS} tasks (Indices 0-${ARRAY_LIMIT})..."

sbatch --array=0-${ARRAY_LIMIT} quartetv2_node.sh
