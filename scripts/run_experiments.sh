#!/bin/bash
#SBATCH -A csds447                         # Course account
#SBATCH -p markov_gpu                      # GPU partition
#SBATCH -C gpul40s                         # Request L40S 48GB GPU
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH -c 4                               # Request 4 CPU cores
#SBATCH --mem=32G                          # Request 32GB system RAM
#SBATCH -t 03:00:00                        # Walltime limit (3 hours)
#SBATCH -J sem_entropy_bench               # Job name
#SBATCH -o %x-%j.out                       # Standard output log
#SBATCH -e %x-%j.err                       # Standard error log

# ==============================================================================
# Environment Configuration
# ==============================================================================
COURSE_DIR="/mnt/vstor/courses/csds447/$USER"

# Set Hugging Face cache directory to course space
export HF_HOME="$COURSE_DIR/hf_cache"

# Activate Virtual Environment
source "$COURSE_DIR/venv/bin/activate"

# Print GPU information for verification
echo "=== GPU Information ==="
nvidia-smi

echo "=== Running Semantic Entropy Experiments ==="

# Define models to evaluate
MODELS=("llama3.2-1b" "llama3.2-3b" "gemma-e2b" "gemma-e4b")

# Run each model on the QASPER dataset
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================================"
    echo "Starting evaluation for model: $MODEL"
    echo "========================================================"
    
    python evaluate_semantic_entropy.py \
        --model "$MODEL" \
        --num-samples 5 \
        --temperature 0.7 \
        --output-dir "figures"
        
    echo "Finished evaluation for $MODEL"
done

echo "========================================================"
echo "All model evaluations completed successfully"
echo "Check the 'figures/' folder for output plots."
echo "========================================================"