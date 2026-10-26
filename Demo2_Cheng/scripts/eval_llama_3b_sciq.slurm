#!/usr/bin/env bash
#SBATCH --job-name=eval_llama_3b_sciq
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-3%2
#SBATCH --output=outputs/%x-%A_%a.out
#SBATCH --error=outputs/%x-%A_%a.err
#SBATCH --signal=USR1@240

set -euo pipefail
# Must submit from ~/447_demo2, so SLURM logs go to ~/447_demo2/outputs.
ROOT="${SLURM_SUBMIT_DIR:?Submit from the project root}"
[[ -f "$ROOT/evaluate_demo2.py" ]] || { echo 'Missing evaluate_demo2.py in project root' >&2; exit 1; }
cd "$ROOT"
source "$ROOT/setup_env_cxg438.sh"  # sets caches; no installs when sourced
module load Python/3.12.3-GCCcore-13.3.0
module load bzip2/1.0.8-GCCcore-13.3.0
source "$VENV_DIR/bin/activate"
if [[ -z "${HF_TOKEN:-}" && -s "$HF_TOKEN_PATH" ]]; then
    export HF_TOKEN="$(< "$HF_TOKEN_PATH")"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
mkdir -p "$ROOT/results/evaluation/metrics" "$ROOT/results/evaluation/judgments"

# No answer generation: load the MERGED CSV and let this same model judge
# all 5 answers for each of the 4 baselines in its assigned quarter of questions.
srun python -u "$ROOT/evaluate_demo2.py" \
    --family llama \
    --model '3B' \
    --dataset sciq \
    --num-shards 4 \
    --shard-index "${SLURM_ARRAY_TASK_ID:?}" \
    --results-dir "$ROOT/results" \
    --phase both \
    --checkpoint-every 20 \
    "$@"
