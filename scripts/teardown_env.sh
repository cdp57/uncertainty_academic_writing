#!/bin/bash
# ==============================================================================
# Teardown / Undo Setup Script for CSDS 447 Markov Cluster
# ==============================================================================

COURSE_DIR="/mnt/vstor/courses/csds447/$USER"
VENV_DIR="$COURSE_DIR/venv"
HF_CACHE_DIR="$COURSE_DIR/hf_cache"

echo "=============================================================================="
echo "WARNING: This will remove the virtual environment, cached model weights,"
echo "output figures, and Slurm log files."
echo "=============================================================================="
read -p "Are you sure you want to reset the environment? (y/N): " CONFIRM

if [[ "$CONFIRM" != [yY] && "$CONFIRM" != [yY][eE][sS] ]]; then
    echo "Teardown cancelled. No files were deleted."
    exit 0
fi

# 1. Deactivate active virtual environment if currently enabled
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating active virtual environment..."
    deactivate 2>/dev/null || true
fi

# 2. Remove Virtual Environment from course directory
if [ -d "$VENV_DIR" ]; then
    echo "Removing Python virtual environment ($VENV_DIR)..."
    rm -rf "$VENV_DIR"
fi

# 3. Remove Cached Hugging Face Models
if [ -d "$HF_CACHE_DIR" ]; then
    echo "Removing Hugging Face model cache ($HF_CACHE_DIR)..."
    rm -rf "$HF_CACHE_DIR"
fi

# 4. Remove generated figures directory
if [ -d "figures" ]; then
    echo "Removing local figures directory (./figures)..."
    rm -rf "figures"
fi

# 5. Clean up Slurm output logs
echo "Cleaning up Slurm log files (*.out, *.err)..."
rm -f sem_entropy_bench-*.out sem_entropy_bench-*.err

echo ""
echo "=============================================================================="
echo "Reset complete! All environment directories and cached data have been removed."
echo "To re-initialize the environment, run: ./setup_env.sh"
echo "=============================================================================="