#!/bin/bash
#SBATCH -A csds447                         # Course account
#SBATCH -p markov_gpu                      # GPU partition
#SBATCH -C "gpu4090|gpul40s"               # Request 4090 or L40S GPU
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH -c 4                               # Request 4 CPU cores
#SBATCH --mem=32G                          # Request 32GB system RAM
#SBATCH -t 00:30:00                        # Short walltime limit (30 minutes)
#SBATCH -J test_uncertainty_sample         # Job name
#SBATCH -o %x-%j.out                       # Standard output log
#SBATCH -e %x-%j.err                       # Standard error log

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# ==============================================================================
# Environment Configuration
# ==============================================================================
USER_NAME=$(whoami)
COURSE_DIR="/mnt/vstor/courses/csds447/${USER_NAME}"
ENV_FILE="${COURSE_DIR}/.env_vars"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

export HF_HOME="${COURSE_DIR}/hf_cache"
source "${COURSE_DIR}/venv/bin/activate"

echo "========================================================"
echo "GPU Information"
echo "========================================================"
nvidia-smi

OUTPUT_DIR="figures_sample_test"
mkdir -p "$OUTPUT_DIR"

# Reduced test matrix
MODELS=("llama3.2-1b" "qwen3-1.7b")
DATASETS=("qasper" "sciq")
JUDGE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

echo ""
echo "========================================================"
echo "SAMPLE TEST STAGE 1: Generating Biased Questions (5 samples)"
echo "========================================================"
python generate_biased_questions.py \
    --model-id "$JUDGE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --max-samples 5

echo ""
echo "========================================================"
echo "SAMPLE TEST STAGE 2: Evaluating Uncertainty Metrics (5 samples)"
echo "========================================================"
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        SPLIT="validation"
        if [ "$DATASET" == "sciq" ]; then
            SPLIT="test"
        fi

        echo ""
        echo "--> Testing uncertainty evaluation: Model=$MODEL | Dataset=$DATASET"
        
        python evaluate_semantic_entropy_2.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --num-samples 5 \
            --num-runs 1 \
            --max-papers 5 \
            --temperature 0.7 \
            --output-dir "$OUTPUT_DIR"
            
    done
done

echo ""
echo "========================================================"
echo "SAMPLE TEST STAGE 4: Generating Sample Metrics Plot"
echo "========================================================"
python plot_metrics_comparison.py \
    --input-dir "$OUTPUT_DIR" \
    --output-file "${OUTPUT_DIR}/metrics_comparison_sample_plot.png" \
    --error-bar sem

echo "========================================================"
echo "Sample Test Job Completed Successfully!"
echo "Check output logs and folder '${OUTPUT_DIR}/' for test artifacts."
echo "========================================================"