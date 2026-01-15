#!/bin/bash

#SBATCH --account=a140
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/capstor/store/cscs/swissai/a140/containers/megatron.toml
#SBATCH --signal=SIGUSR2@600
#SBATCH --no-requeue

# Stop if any command fails
set -e

echo "START TIME: $(date)"
echo "Running on host: $(hostname)"
echo "Config: MULT=${MULT}, TOKENS=${TOKENS}"

# ==========================================
# 1. Static Environment Setup
# ==========================================

export VOCAB_SIZE=32000 
export BATCH_SIZE=128
export ACC_STEPS=4
export SEQUENCE_LENGTH=512
export DATASET="c4"

# Local Dependencies
export WANDB_ENTITY=ist
cd /capstor/store/cscs/swissai/a140/codebases/backprop-scaling-laws
pip install schedulefree

# Data Paths (Assumed pre-synced or synced by the first job to land)
export DATASET_BUFFER="/iopsstor/scratch/cscs/blacksamorez/datasets"

# Quantization
export W_QUANT="NoQuantizer"
export A_QUANT="NoQuantizer"
export W_QUANT_KWARGS="{}"
export A_QUANT_KWARGS="{}"

# ==========================================
# 2. Calculation & Execution
# ==========================================

export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
export WARMUP_STEPS=$((ITERATIONS / 10))

# WandB Prefix
WANDB_PREFIX="${MODEL_SIZE_PREFIX}-BF16:BF16-${MULT}x-${DATASET}"

echo "Launching torchrun..."

torchrun --nproc_per_node=4 ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --datasets-dir $DATASET_BUFFER \
    --latest-ckpt-interval 1000 \
    --model llama \
    --vocab-size $VOCAB_SIZE \
    --compile \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --wandb \
    --wandb-project "backprop-scaling-laws" \
    --wandb-run-prefix "${WANDB_PREFIX}" \
    --log-interval 1 \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --warmup-steps ${WARMUP_STEPS} \
    --iterations ${ITERATIONS} \
    --lr ${LR} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --a-quant-kwargs "${A_QUANT_KWARGS}"

echo "END TIME: $(date)"
