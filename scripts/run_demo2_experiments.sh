#!/bin/bash
#SBATCH -A csds447                         # Course account
#SBATCH -p markov_gpu                      # GPU partition
#SBATCH -C "gpu4090|gpul40s"               # Request 4090 or L40S 48GB GPU
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH -c 4                               # Request 4 CPU cores
#SBATCH --mem=32G                          # Request 32GB system RAM
#SBATCH -t 12:00:00                        # Walltime limit (12 hours for full suite)
#SBATCH -J full_uncertainty_suite          # Job name
#SBATCH -o %x-%j.out                       # Standard output log
#SBATCH -e %x-%j.err                       # Standard error log

# Force unbuffered Python output for real-time Slurm logging
export PYTHONUNBUFFERED=1

# ==============================================================================
# Environment Configuration
# ==============================================================================
USER_NAME=$(whoami)
COURSE_DIR="/mnt/vstor/courses/csds447/${USER_NAME}"
ENV_FILE="${COURSE_DIR}/.env_vars"

# Load saved Hugging Face token and environment variables if present
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Set Hugging Face cache directory to course space
export HF_HOME="${COURSE_DIR}/hf_cache"

# Activate Virtual Environment
source "${COURSE_DIR}/venv/bin/activate"

# Verify GPU
echo "========================================================"
echo "GPU Information"
echo "========================================================"
nvidia-smi

DATASET_DIR="data"
RESULTS_DIR="results"
FIGURES_DIR="figures"

mkdir -p "$DATASET_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$FIGURES_DIR"

# Models and Datasets matrix
MODELS=("llama3.2-1b" "llama3.2-3b" "qwen3-1.7b" "qwen3-4b")
DATASETS=("qasper" "sciq")
JUDGE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

echo ""
echo "========================================================"
echo "STAGE 1: Generating Biased Questions (Confirmation Bias)"
echo "========================================================"
python generate_biased_questions.py \
    --model-id "$JUDGE_MODEL" \
    --output-dir "$DATASET_DIR"

echo ""
echo "========================================================"
echo "STAGE 2: Evaluating Uncertainty Metrics across Suite"
echo "========================================================"
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        # Determine appropriate split
        SPLIT="validation"
        if [ "$DATASET" == "sciq" ]; then
            SPLIT="test"
        fi

        echo ""
        echo "--> Running uncertainty evaluation: Model=$MODEL | Dataset=$DATASET ($SPLIT)"
        
        python evaluate_semantic_entropy_2.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --num-samples 5 \
            --num-runs 3 \
            --temperature 0.7 \
            --output-dir "$RESULTS_DIR"
            
    done
done



echo ""
echo "========================================================"
echo "STAGE 3: Generating Metrics Comparison Plot"
echo "========================================================"
python plot_metrics_comparison.py \
    --input-dir "$RESULTS_DIR" \
    --output-file "${FIGURES_DIR}/metrics_comparison_plot.png" \
    --error-bar sem

echo "========================================================"
echo "Full Experimental Suite Completed Successfully!"
echo "Outputs and plots saved to: ${FIGURES_DIR}/"
echo "========================================================"