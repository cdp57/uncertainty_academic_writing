#!/bin/bash
#SBATCH -A csds447                          # Course account
#SBATCH -p markov_gpu                       # Partition
#SBATCH -c 4                                # Request 4 CPU cores
#SBATCH --mem=16G                           # Request 16GB system RAM
#SBATCH -t 00:30:00                         # Walltime limit (30 minutes)
#SBATCH -J plot_uncertainty_metrics         # Job name
#SBATCH -o logs/%x-%j.out                   # Standard output log
#SBATCH -e logs/%x-%j.err                   # Standard error log

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# ==============================================================================
# Environment Configuration
# ==============================================================================
USER_NAME=$(whoami)
COURSE_DIR="/mnt/vstor/courses/csds447/${USER_NAME}"

# Activate Virtual Environment
source "${COURSE_DIR}/venv/bin/activate"

echo "========================================================"
echo "Starting AUROC & AURC Plot Generation"
echo "========================================================"

# OPTION 1: If your CSVs are stored in separate directories
python plot_metrics_full.py \
    --entropy-dir "results" \
    --confidence-dir "../Demo2_Cheng/results" \
    --evaluated-dir "results" \
    --output-dir "figures"

echo "========================================================"
echo "Plotting completed! Figures saved to figures/"
echo "========================================================"