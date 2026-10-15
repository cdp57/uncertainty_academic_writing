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

# Set up HF login
echo ""
echo "=============================================================================="
echo "Hugging Face Authentication"
echo "=============================================================================="
read -sp "Paste your Hugging Face Access Token (input hidden): " HF_TOKEN
echo ""

if [ -n "$HF_TOKEN" ]; then
    echo "Authenticating with Hugging Face..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
    
    # Save environment variables for Slurm batch jobs
    echo "export HF_HOME=\"$HF_CACHE_DIR\"" > "$ENV_FILE"
    echo "export HF_TOKEN=\"$HF_TOKEN\"" >> "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    
    echo "Authentication successful! Variables saved to: $ENV_FILE"
else
    echo "Warning: No token provided. You will need to set HF_TOKEN manually to download gated models."
fi

echo ""
echo "=============================================================================="
echo "Setup Complete"
echo "Virtual environment created at: $VENV_DIR"
echo "Hugging Face cache set to:      $HF_CACHE_DIR"
echo "=============================================================================="