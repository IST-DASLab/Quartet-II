#!/bin/bash
set -e

# common env
export VOCAB_SIZE=32000
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512
export DATASET="c4"

# 1. Define Model Configurations
MODEL_SIZES=(
  "30M"
  "50M"
  # "100M"
  # "200M"
)

# 2. Define Quantization Setup Pairs (HADAMARD_DIM:SCALE_DTYPE)
QUANT_SETUPS=(
  "128;fp32;no"
  "128;fp32;sr"
  "128;fp32;eden"
  "16;e4m3;no"
  "16;e4m3;sr"
  "16;e4m3;eden"
  "32;e8m0;no"
  "32;e8m0;sr"
  "32;e8m0;eden"
)

for SIZE in "${MODEL_SIZES[@]}"; do
  if [ "$SIZE" == "30M" ]; then
    export N_LAYER=6; export N_EMBD=640; export N_HEAD=5; export LR=0.0012; export BASE_TOKENS=3000000000
  elif [ "$SIZE" == "50M" ]; then
    export N_LAYER=7; export N_EMBD=768; export N_HEAD=6; export LR=0.0012; export BASE_TOKENS=5000000000
  elif [ "$SIZE" == "100M" ]; then
    export N_LAYER=8; export N_EMBD=1024; export N_HEAD=8; export LR=0.0006; export BASE_TOKENS=10000000000
  elif [ "$SIZE" == "200M" ]; then
    export N_LAYER=10; export N_EMBD=1280; export N_HEAD=10; export LR=0.0003; export BASE_TOKENS=20000000000
  fi
  export MODEL_SIZE_PREFIX=$SIZE

  TOKENS_LIST=(
    $((BASE_TOKENS/4))
    $((BASE_TOKENS/2))
    ${BASE_TOKENS}
    $((BASE_TOKENS*2))
    $((BASE_TOKENS*4))
    $((BASE_TOKENS*8))
  )

  for TOKENS in "${TOKENS_LIST[@]}"; do
    export TOKENS
    export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
    export WARMUP_STEPS=$((ITERATIONS / 10))

    for SETUP in "${QUANT_SETUPS[@]}"; do
      export HADAMARD_DIM=$(echo $SETUP | cut -d\; -f1)
      export SCALE_DTYPE=$(echo $SETUP | cut -d\; -f2)
      export UNBIASED=$(echo $SETUP | cut -d\; -f3)

      export W_QUANT="NoQuantizer"
      export W_BITS=16
      export W_QUANT_KWARGS="{}"
      export A_QUANT="NoQuantizer"
      export A_BITS=16
      export A_QUANT_KWARGS="{}"
      export G_QUANT="EdenSRQuantizer"
      export G_BITS=4
      export G_QUANT_KWARGS="{\"hadamard_dim\": ${HADAMARD_DIM}, \"rerotate\": \"signs\", \"scale_dtype\": \"${SCALE_DTYPE}\", \"unbiased\": \"${UNBIASED}\"}"
      export BACKWARD_SCHEME="Q(E)Q(Wt)t_Q(Et)Q(Xt)t"
      export BACKWARD_SCHEME_KWARGS="{}"

      WANDB_PREFIX="${MODEL_SIZE_PREFIX}-TOK${TOKENS}-${G_QUANT}@${G_BITS}@${SETUP}-${DATASET}"

      echo "========================================================================="
      echo "RUNNING: Model=${MODEL_SIZE_PREFIX}, Tokens=${TOKENS}, Dim=${HADAMARD_DIM}, Scale=${SCALE_DTYPE}, Unbiased=${UNBIASED}"
      echo "iters=${ITERATIONS}, warmup=${WARMUP_STEPS}"
      echo "========================================================================="

      TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
      CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
      torchrun --nproc_per_node=8 ./src/main.py \
        --distributed-backend nccl \
        --dataset ${DATASET} \
        --datasets-dir /dev/shm/datasets \
        --latest-ckpt-interval 1000 \
        --model llama \
        --compile \
        --acc-steps ${ACC_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --wandb \
        --wandb-project "backward-laws" \
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
        --a-quant-kwargs "${A_QUANT_KWARGS}" \
        --g-quant ${G_QUANT} \
        --g-quant-kwargs "${G_QUANT_KWARGS}" \
        --backward-scheme ${BACKWARD_SCHEME} \
        --backward-scheme-kwargs "${BACKWARD_SCHEME_KWARGS}"
    done
  done
done
