#!/bin/bash
#SBATCH -A csds447                          # Course account
#SBATCH -p markov_gpu                       # GPU partition
#SBATCH -C "gpu4090|gpul40s"                # Request 4090 or L40S GPU
#SBATCH -x class21t                         # Exclude faulty CUDA driver node
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH -c 4                                # Request 4 CPU cores
#SBATCH --mem=48G                           # Request 48GB system RAM
#SBATCH -t 12:00:00                         # Walltime limit for full suite (12 hours)
#SBATCH -J full_llm_judge_suite             # Job name
#SBATCH -o logs/%x-%j.out                   # Standard output log
#SBATCH -e logs/%x-%j.err                   # Standard error log

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================================
# Environment Configuration
# ==============================================================================
USER_NAME=$(whoami)
COURSE_DIR="/mnt/vstor/courses/csds447/${USER_NAME}"
ENV_FILE="${COURSE_DIR}/.env_vars"

# Load saved environment variables and HF token if available
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Set Hugging Face cache directory to course space
export HF_HOME="${COURSE_DIR}/hf_cache"

# Activate Virtual Environment
source "${COURSE_DIR}/venv/bin/activate"

# Matrix definition matching experimental setup
MODELS=("llama3.2_1b" "llama3.2_3b" "qwen3_1.7b" "qwen3_4b")
DATASETS=("qasper" "sciq")
JUDGE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
RESULTS_DIR="results"

echo "========================================================"
echo "Starting Full LLM-as-a-Judge Evaluation Suite"
echo "Evaluator Model: $JUDGE_MODEL"
echo "========================================================"
nvidia-smi

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do

        # Determine appropriate dataset split
        SPLIT="validation"
        if [ "$DATASET" == "sciq" ]; then
            SPLIT="test"
        fi

        # Find matching results CSV from Stage 2 output
        # Checks common filename conventions (e.g., results_sciq_llama3.2-1b.csv)
        TARGET_CSV="${RESULTS_DIR}/results_${DATASET}_${MODEL}.csv"
        
        if [ ! -f "$TARGET_CSV" ]; then
            # Fallback search if filenames use different separator conventions
            TARGET_CSV=$(ls ${RESULTS_DIR}/*${DATASET}*${MODEL}*.csv 2>/dev/null | head -n 1)
        fi

        if [ -z "$TARGET_CSV" ] || [ ! -f "$TARGET_CSV" ]; then
            echo "[WARNING] Output file for Model=$MODEL | Dataset=$DATASET not found in $RESULTS_DIR. Skipping..."
            continue
        fi

        echo ""
        echo "========================================================"
        echo "--> Running Judge: Model=$MODEL | Dataset=$DATASET ($SPLIT)"
        echo "    Input CSV: $TARGET_CSV"
        echo "========================================================"

        python LLM_judge.py \
            --results-csv "$TARGET_CSV" \
            --judge-model "$JUDGE_MODEL" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --output-dir "$RESULTS_DIR"

    done
done

echo ""
echo "========================================================"
echo "Full LLM Judge Evaluation Suite Completed Successfully!"
echo "Evaluated outputs saved to: ${RESULTS_DIR}/"
echo "========================================================"