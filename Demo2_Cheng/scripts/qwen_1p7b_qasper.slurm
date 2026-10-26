#!/usr/bin/env bash
#SBATCH --job-name=qwen_1p7b_qasper
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
# Submit from ~/447_demo2 (NOT scripts/): sbatch scripts/qwen_1p7b_qasper.slurm
ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
if [[ ! -f "$ROOT/qwen_qasper.py" ]]; then
    echo 'Submit from ~/447_demo2; Python entrypoint not found.' >&2
    exit 1
fi
cd "$ROOT"
source "$ROOT/setup_env_cxg438.sh"  # cache paths only, no installs
module load Python/3.12.3-GCCcore-13.3.0
module load bzip2/1.0.8-GCCcore-13.3.0
source "$VENV_DIR/bin/activate"
if [[ -z "${HF_TOKEN:-}" && -s "$HF_TOKEN_PATH" ]]; then
    export HF_TOKEN="$(< "$HF_TOKEN_PATH")"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

# Four question-index shards only; each question itself uses five distinct seeds.
TASK_ID="${SLURM_ARRAY_TASK_ID:?Array task ID required}"
SHARD_INDEX="$TASK_ID"

printf 'Config=%s; sample seeds=42,123,2026,3407,5678; shard=%s/4\n' "qwen_1p7b_qasper" "$SHARD_INDEX"
printf 'Source/results=%s; venv=%s; HF cache=%s\n' "$ROOT" "$VENV_DIR" "$HF_HOME"
python - <<'PY_CHECK'
import torch
print('torch', torch.__version__, 'CUDA build', torch.version.cuda, 'available', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; check SLURM GPU allocation or PyTorch wheel')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY_CHECK

srun python -u "$ROOT/qwen_qasper.py" \
    --model '1.7B' \
    --num-shards 4 \
    --shard-index "$SHARD_INDEX" \
    --checkpoint-every 5 \
    --results-dir "$ROOT/results" \
    "$@"
