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
#SBATCH --output=/capstor/store/cscs/swissai/a140/codebases/backprop-scaling-laws/logs/slurm/quartetv2_sweep/bw-sweep-%A_%a.out
#SBATCH --error=/capstor/store/cscs/swissai/a140/codebases/backprop-scaling-laws/logs/slurm/quartetv2_sweep/bw-sweep-%A_%a.err

# Stop if any command fails and on unbound variables
set -eu

# ==========================================
# 0. Configuration & Array Mapping
# ==========================================

MODEL_CONFIGS=($STR_MODEL_CONFIGS)
MULTIPLIERS=($STR_MULTIPLIERS)
QUANT_SETUPS=($STR_QUANT_SETUPS)

N_MODELS=${#MODEL_CONFIGS[@]}
N_MULTS=${#MULTIPLIERS[@]}
N_QUANTS=${#QUANT_SETUPS[@]}

# Decode SLURM_ARRAY_TASK_ID to get indices for each dimension
# Index logic: Model -> Multiplier -> Quant Setup
idx_quant=$(( SLURM_ARRAY_TASK_ID % N_QUANTS ))
idx_mult=$(( (SLURM_ARRAY_TASK_ID / N_QUANTS) % N_MULTS ))
idx_model=$(( SLURM_ARRAY_TASK_ID / (N_QUANTS * N_MULTS) ))

# Select specific parameters based on calculated indices
CURRENT_MODEL_CFG="${MODEL_CONFIGS[$idx_model]}"
CURRENT_MULT="${MULTIPLIERS[$idx_mult]}"
CURRENT_SETUP="${QUANT_SETUPS[$idx_quant]}"

# Parse Model Config
IFS=":" read -r MODEL_SIZE_PREFIX N_LAYER N_EMBD N_HEAD LR BASE_TOKENS <<< "$CURRENT_MODEL_CFG"

# Calculate Tokens (using python for float math)
TOKENS=$(python3 -c "print(int($BASE_TOKENS * $CURRENT_MULT))")

# Parse Quant Setup
IFS=":" read -r HADAMARD_DIM DELAYED_AMAX DISABLE_FORWARD_QUANT DISABLE_BACKWARD_QUANT FOUR_OVER_SIX <<< "$CURRENT_SETUP"

# ==========================================
# 1. Static Environment Setup
# ==========================================

echo "START TIME: $(date)"
echo "Running on host: $(hostname)"
echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Config: ${MODEL_SIZE_PREFIX} | Multiplier: ${CURRENT_MULT} | Tokens: ${TOKENS}"
echo "Scheme: Hadamard DIM=${HADAMARD_DIM}, delayed amax=${DELAYED_AMAX}"

export VOCAB_SIZE=32000 
export BATCH_SIZE=128
export ACC_STEPS=4
export SEQUENCE_LENGTH=512
export DATASET="c4"
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export WANDB_ENTITY=ist

cd /capstor/store/cscs/swissai/a140/codebases/backprop-scaling-laws
pip install schedulefree
export DATASET_BUFFER="/iopsstor/scratch/cscs/blacksamorez/datasets"

# ==========================================
# 2. Quantization Configuration
# ==========================================

# Special
export SPECIAL_SCHEME="quartet_v2"
export SPECIAL_SCHEME_KWARGS="{\"hadamard_dim\": $HADAMARD_DIM, \"delayed_amax\": $DELAYED_AMAX, \"disable_forward_quant\": ${DISABLE_FORWARD_QUANT}, \"disable_backward_quant\": ${DISABLE_BACKWARD_QUANT}, \"four_over_six\": ${FOUR_OVER_SIX}}"

# ==========================================
# 3. Calculation & Execution
# ==========================================

export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
export WARMUP_STEPS=$((ITERATIONS / 10))

# WandB Prefix
SETUP_STR="${HADAMARD_DIM};${DELAYED_AMAX};${DISABLE_FORWARD_QUANT};${DISABLE_BACKWARD_QUANT}"
if [ "${FOUR_OVER_SIX}" = "true" ]; then
    SETUP_STR="${SETUP_STR};${FOUR_OVER_SIX}"
fi
WANDB_PREFIX="${MODEL_SIZE_PREFIX}-TOK${TOKENS}-${SPECIAL_SCHEME}@${SETUP_STR}-${DATASET}"

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
    --special-scheme ${SPECIAL_SCHEME} \
    --special-scheme-kwargs "${SPECIAL_SCHEME_KWARGS}"

echo "END TIME: $(date)"
