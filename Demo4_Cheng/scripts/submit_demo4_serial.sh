#!/usr/bin/env bash
# CPU preflight + four single-GPU dependent jobs, one at a time.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs results/demo4/responses results/demo4/csv
# Validate all 3005 frozen Demo 3 inputs without loading models or a GPU.
# Runs via the existing project's Python environment.
module load Python/3.12.3-GCCcore-13.3.0
source "$ROOT/setup_env_cxg438.sh"
source "$VENV_DIR/bin/activate"
python demo4_collect.py --model qwen_1p7b --validate-only
last=''
for model in qwen_1p7b qwen_4b llama_1b llama_3b; do
  options=(--parsable)
  if [[ -n "$last" ]]; then options+=("--dependency=afterok:$last"); fi
  jobid="$(sbatch "${options[@]}" "scripts/demo4_${model}.slurm")"
  echo "Submitted ${model}: job ${jobid}${last:+ after successful job $last}"
  last="$jobid"
done
cat <<'EOF'
IMPORTANT: Do not also submit manually. Only one of these four jobs can run.
If one job times out/fails, later jobs are blocked by afterok dependencies.
Resume the failed model using its Slurm script, then schedule remaining models.
Monitor: python demo4_status.py; squeue --me
EOF
