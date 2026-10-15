#!/bin/bash
# ==============================================================================
# Environment Setup Script for CSDS 447 Markov Cluster
# ==============================================================================

# Define storage directories in course space
COURSE_DIR="/mnt/vstor/courses/csds447/$USER"
VENV_DIR="$COURSE_DIR/venv"
HF_CACHE_DIR="$COURSE_DIR/hf_cache"

echo "Creating directories in course space: $COURSE_DIR..."
mkdir -p -m 700 "$COURSE_DIR"
mkdir -p "$HF_CACHE_DIR"
mkdir -p "figures"

# Export Hugging Face cache directory to environment variable
export HF_HOME="$HF_CACHE_DIR"

# Create and activate Python virtual environment
echo "Setting up Python virtual environment"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install required dependencies
echo "Installing dependencies"
pip install --upgrade pip
pip install -r requirements_hpc.txt

echo ""
echo "=============================================================================="
echo "Setup Complete"
echo "Virtual environment created at: $VENV_DIR"
echo "Hugging Face cache set to:      $HF_CACHE_DIR"
echo "=============================================================================="