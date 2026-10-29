#!/bin/bash
#SBATCH -A csds447                          # Course account
#SBATCH -p markov_gpu                       # GPU partition
#SBATCH -C "gpu4090|gpul40s"                # GPU request
#SBATCH -x class21t                         # Exclude faulty node
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH -c 4                                # 4 CPU cores
#SBATCH --mem=48G                           # 48GB system RAM
#SBATCH -t 04:00:00                         # 4 hour walltime limit
#SBATCH -J sciq_full_pipeline               # Job name
#SBATCH -o logs/%x-%j.out                   # Stdout log
#SBATCH -e logs/%x-%j.err                   # Stderr log

# Environment & Memory Configuration
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USER_NAME=$(whoami)
COURSE_DIR="/mnt/vstor/courses/csds447/${USER_NAME}"
ENV_FILE="${COURSE_DIR}/.env_vars"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

export HF_HOME="${COURSE_DIR}/hf_cache"
source "${COURSE_DIR}/venv/bin/activate"

# Directory Structure
ENTROPY_DIR="results/biased/entropy"
CONFIDENCE_DIR="results/biased/confidence"
EVALUATED_DIR="results/biased/evaluation"
FIGURES_DIR="figures/biased"
DATA_DIR="data"

MODELS=("llama3.2_1b" "llama3.2_3b" "qwen3_1.7b" "qwen3_4b")
JUDGE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATASET="sciq"
SPLIT="test"  # Standard SciQ split

echo "========================================================"
echo "Starting SciQ End-to-End Uncertainty & Accuracy Pipeline"
echo "========================================================"
nvidia-smi

# ==============================================================================
# STAGE 1 & 2: Entropy Generation & LLM Judge Evaluation
# ==============================================================================
for MODEL in "${MODELS[@]}"; do
    CLEAN_MODEL=$(echo "$MODEL" | tr '/' '_' | tr '-' '_')
    ENTROPY_CSV="${ENTROPY_DIR}/results_${DATASET}_${CLEAN_MODEL}.csv"

    echo ""
    echo "--------------------------------------------------------"
    echo "--> [Stage 1] Generating Entropy: Model=$MODEL | Dataset=$DATASET"
    echo "--------------------------------------------------------"
    
    python evaluate_biased_semantic_entropy.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --num-samples 10 \
        --temperature 0.7 \
        --biased-input-csv "${DATA_DIR}/sciq_biased_questions.csv" \
        --output-dir "$ENTROPY_DIR"

    if [ ! -f "$ENTROPY_CSV" ]; then
        echo "[ERROR] Entropy output file not found: $ENTROPY_CSV. Skipping judge for $MODEL..."
        continue
    fi

    echo ""
    echo "--------------------------------------------------------"
    echo "--> [Stage 2] Running LLM Judge: Model=$MODEL | File=$ENTROPY_CSV"
    echo "--------------------------------------------------------"
    
    python LLM_judge_biased.py \
        --results-csv "$ENTROPY_CSV" \
        --judge-model "$JUDGE_MODEL" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --output-dir "$EVALUATED_DIR"

done

# ==============================================================================
# STAGE 3: Plot Metric Comparison Curves (AUROC & AURC)
# ==============================================================================
echo ""
echo "========================================================"
echo "--> [Stage 3] Generating AUROC and AURC Comparison Plots"
echo "========================================================"

python plot_metrics_biased.py \
    --entropy-dir "$ENTROPY_DIR" \
    --confidence-dir "$CONFIDENCE_DIR" \
    --evaluated-dir "$EVALUATED_DIR" \
    --output-dir "$FIGURES_DIR"

echo ""
echo "========================================================"
echo "SciQ End-to-End Pipeline Execution Complete!"
echo "Plots saved to: ${FIGURES_DIR}/"
echo "========================================================"