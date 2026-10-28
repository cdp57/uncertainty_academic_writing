#!/usr/bin/env bash
#SBATCH --job-name=bias_collect_llama_1b
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-15%2
#SBATCH --output=outputs/%x-%A_%a.out
#SBATCH --error=outputs/%x-%A_%a.err
#SBATCH --signal=USR1@240

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?Submit from ~/447_demo2}"
cd "$ROOT"
[[ -f "$ROOT/collect_bias_sciq.py" ]] || { echo 'Missing root Python program' >&2; exit 1; }
[[ -f "$ROOT/sciq_biased_questions_cleaned.csv" ]] || { echo 'Missing teammate CSV' >&2; exit 1; }
[[ -d "$ROOT/outputs" ]] || { echo 'Create outputs/ before sbatch' >&2; exit 1; }
source "$ROOT/setup_env_cxg438.sh"  # exports cache paths; NO installation
module load Python/3.12.3-GCCcore-13.3.0
module load bzip2/1.0.8-GCCcore-13.3.0
source "$VENV_DIR/bin/activate"
if [[ -z "${HF_TOKEN:-}" && -s "$HF_TOKEN_PATH" ]]; then
  export HF_TOKEN="$(< "$HF_TOKEN_PATH")"
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
SHARD="${SLURM_ARRAY_TASK_ID:?Must submit as a job array}"
echo "Model=llama/1B stage=collect original shard=$SHARD/16; 2 bias conditions per original question"
python -u - <<'PYCUDA'
import torch
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU ready:',torch.cuda.is_available(),flush=True)
assert torch.cuda.is_available(), 'Allocated GPU not available; check driver/PyTorch build'
print('GPU:',torch.cuda.get_device_name(0),flush=True)
PYCUDA

srun python -u "$ROOT/collect_bias_sciq.py" \
  --family 'llama' \
  --model '1B' \
  --num-shards 16 \
  --shard-index "$SHARD" \
  --checkpoint-every 4 \
  --data-csv "$ROOT/sciq_biased_questions_cleaned.csv" \
  --results-dir "$ROOT/results" \
  "$@"
