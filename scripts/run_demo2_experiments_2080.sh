#!/bin/bash
#SBATCH -A csds447                          # Course account
#SBATCH -p markov_gpu                       # GPU partition
#SBATCH -C gpu2080                          # Request RTX 2080 GPUs
#SBATCH -x class21t                         # Exclude faulty CUDA driver node
#SBATCH --gres=gpu:2                        # Request 4x RTX 2080 GPUs
#SBATCH -c 4                                # Request 8 CPU cores (4 cores per process pair)
#SBATCH --mem=48                           # Request 48GB system RAM
#SBATCH -t 14:00:00                         # Walltime limit
#SBATCH -J pipeline_4x2080_parallel         # Job name
#SBATCH -o logs/%x-%j.out                   # Standard output log
#SBATCH -e logs/%x-%j.err                   # Standard error log

# Force unbuffered Python output for real-time Slurm logging
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

# Ensure output directories exist
mkdir -p logs data results figures

echo "========================================================"
echo "GPU Information (4x RTX 2080)"
echo "========================================================"
nvidia-smi

# Target configuration matrix
MODELS=("llama3.2-1b" "llama3.2-3b" "qwen3-1.7b" "qwen3-4b")
GEN_MODEL="meta-llama/Llama-3.2-3B-Instruct"

echo ""
echo "========================================================"
echo "STAGE 1: Parallel Question Generation (2x2 GPU Split)"
echo "========================================================"
# SciQ on GPU Pair 0 (GPUs 0,1)
CUDA_VISIBLE_DEVICES=0,1 python generate_biased_questions.py \
    --model-id "$GEN_MODEL" \
    --dataset "sciq" \
    --output-dir "data" &

# QASPER on GPU Pair 1 (GPUs 2,3)
CUDA_VISIBLE_DEVICES=2,3 python generate_biased_questions.py \
    --model-id "$GEN_MODEL" \
    --dataset "qasper" \
    --output-dir "data" &

# Wait for both question generation tasks to complete
wait

echo ""
echo "========================================================"
echo "STAGE 2: Evaluating Uncertainty Metrics (2x2 Parallel)"
echo "========================================================"
for MODEL in "${MODELS[@]}"; do
    echo "--------------------------------------------------------"
    echo "Launching parallel evaluation for Model: $MODEL"
    echo "--------------------------------------------------------"

    # Job A: SciQ on GPUs 0 and 1
    CUDA_VISIBLE_DEVICES=0,1 python evaluate_semantic_entropy_2.py \
        --model "$MODEL" \
        --dataset "sciq" \
        --split "test" \
        --num-samples 5 \
        --num-runs 1 \
        --temperature 0.7 \
        --output-dir "results" &

    # Job B: QASPER on GPUs 2 and 3
    CUDA_VISIBLE_DEVICES=2,3 python evaluate_semantic_entropy_2.py \
        --model "$MODEL" \
        --dataset "qasper" \
        --split "validation" \
        --num-samples 5 \
        --num-runs 1 \
        --temperature 0.7 \
        --output-dir "results" &

    # Wait for both datasets to complete before loading the next model
    wait
    echo "Completed evaluation for $MODEL across both datasets."
done

echo ""
echo "========================================================"
echo "STAGE 3: Generating Uncertainty Metric Comparison Plots"
echo "========================================================"
python plot_metrics_comparison.py \
    --input-dir "results" \
    --output-file "figures/uncertainty_metrics_comparison_4x2080.png" \
    --error-bar sem

echo "========================================================"
echo "Full 4x2080 Parallel Pipeline Completed Successfully!"
echo "Outputs and plots saved to: figures/uncertainty_metrics_comparison_4x2080.png"
echo "========================================================"