#!/usr/bin/env bash
#SBATCH --job-name=d3_llama_3b
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err
#SBATCH --signal=USR1@240

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?Submit from project root}"
cd "$ROOT"
[[ -f "$ROOT/demo3_collect.py" ]] || { echo 'Submit from 447_demo2 root' >&2; exit 1; }
# Source cache settings; installations happen ONLY when setup script is run directly.
source "$ROOT/setup_env_cxg438.sh"
module load Python/3.12.3-GCCcore-13.3.0
source "$VENV_DIR/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
# Fail fast if snapshots were not prefetched; do not download while holding a GPU.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Only ONE process, ONE GPU, ONE sample per method.
srun python -u "$ROOT/demo3_collect.py" --model 'llama_3b' --dataset all --quantization 4bit "$@"
