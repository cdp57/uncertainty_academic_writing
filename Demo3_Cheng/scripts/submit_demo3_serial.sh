#!/usr/bin/env bash
# Submit four GPU jobs with afterok dependencies: at most one running at a time.
# Run from 447_demo2 root only after CPU preparation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs
python demo3_prepare.py --validate-only
prev=''
for model in llama_1b llama_3b qwen_1p7b qwen_4b; do
  args=(--parsable)
  if [[ -n "$prev" ]]; then args+=(--dependency="afterok:$prev"); fi
  id="$(sbatch "${args[@]}" "scripts/demo3_${model}.slurm")"
  echo "Scheduled $model as job $id${prev:+ after $prev succeeds}"
  prev="$id"
done
printf '\nWARNING: If a job hits its walltime before completion, its downstream\n'
printf 'dependencies will not run. Resume the incomplete model first, then\n'
printf 'submit the remaining models sequentially. Check with: python demo3_status.py\n'
