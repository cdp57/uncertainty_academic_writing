#!/usr/bin/env bash
#SBATCH --job-name=d4_qwen_4b
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
ROOT="${SLURM_SUBMIT_DIR:?Submit from the 447_demo2 root}"
cd "$ROOT"
[[ -f "$ROOT/demo4_collect.py" ]] || { echo 'demo4_collect.py missing in project root' >&2; exit 1; }
# Identical course-storage venv/HF cache as Demo 3. Do not install in GPU job.
module load Python/3.12.3-GCCcore-13.3.0
source "$ROOT/setup_env_cxg438.sh"
source "$VENV_DIR/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Exactly one process, one GPU, no array, no batching or extra seeds.
srun python -u "$ROOT/demo4_collect.py" --model qwen_4b --dataset all --quantization 4bit "$@"
