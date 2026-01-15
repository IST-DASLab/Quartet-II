#!/bin/bash

# Configuration Arrays

# Model Config Format: "MODEL_SIZE_PREFIX:N_LAYER:N_EMBD:N_HEAD:LR:BASE_TOKENS"
MODEL_CONFIGS=(
    # "30M:6:640:5:0.0012:3000000000"
    # "50M:7:768:6:0.0012:5000000000"
    # "100M:8:1024:8:0.0009:10000000000"
    "200M:10:1280:10:0.00072:20000000000"
)
MULTIPLIERS=(
    0.25
    0.5
    1
    # 2
    # 4
    # 8
)

# Directory for logs
LOG_DIR="/capstor/store/cscs/swissai/a140/codebases/backprop-scaling-laws/logs/slurm/baseline"
mkdir -p "$LOG_DIR"

for config in "${MODEL_CONFIGS[@]}"; do
    # Parse configuration string into variables
    IFS=":" read -r MODEL_SIZE_PREFIX N_LAYER N_EMBD N_HEAD LR BASE_TOKENS <<< "$config"
    
    # Export variables so they are available to sbatch (via --export=ALL)
    export MODEL_SIZE_PREFIX N_LAYER N_EMBD N_HEAD LR BASE_TOKENS

    for mult in "${MULTIPLIERS[@]}"; do
        
        # 1. Calculate Tokens (Python is safest for float math)
        tokens=$(python3 -c "print(int($BASE_TOKENS * $mult))")
        
        # 2. Create a unique Job Name
        job_name="tribe-${MODEL_SIZE_PREFIX::-1}m-baseline-m${mult}"
        
        echo "Submitting: Mult=${mult} | Tokens=${tokens}"
        
        # 3. Submit to Slurm
        # We pass variables using --export. ALL ensures current env vars pass too.
        # We specifically override BITS, MULT, TOKENS.
        
        sbatch \
            --job-name="${job_name}" \
            --output="${LOG_DIR}/${job_name}-%j.out" \
            --error="${LOG_DIR}/${job_name}-%j.err" \
            --export=ALL,MULT=${mult},TOKENS=${tokens} \
            baseline_node.sh
    done
done
