#!/usr/bin/env bash
# Source for cache environment, execute once to install Python dependencies.
# Code, output logs, CSV/XLSX all remain in ~/447_demo2.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  umask 077
fi
COURSE_DIR="/mnt/vstor/courses/csds447/${USER:?USER missing}"
VENV_DIR="$COURSE_DIR/.venv"
HF_CACHE_DIR="$COURSE_DIR/hf_cache"
CACHE_DIR="$COURSE_DIR/cache"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HF_XET_CACHE="$HF_CACHE_DIR/xet"
export HF_ASSETS_CACHE="$HF_CACHE_DIR/assets"
export HF_TOKEN_PATH="$HF_CACHE_DIR/token"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export UV_CACHE_DIR="$CACHE_DIR/uv"
export TORCH_HOME="$CACHE_DIR/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export CUDA_CACHE_PATH="$CACHE_DIR/cuda/ComputeCache"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export NUMBA_CACHE_DIR="$CACHE_DIR/numba"
export PYTHONPYCACHEPREFIX="$CACHE_DIR/pycache"
export TMPDIR="$CACHE_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

mkdir -p "$COURSE_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$(dirname "$CUDA_CACHE_PATH")" \
  "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$TMPDIR"
# Keep code/outputs in HOME. SLURM output targets require these directories already.
mkdir -p "$PROJECT_DIR/outputs" "$PROJECT_DIR/results"

# Called with `source` inside each SLURM job: only environment variables/dirs.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

module load Python/3.12.3-GCCcore-13.3.0
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
# CUDA 12.8 matches user's Markov NVIDIA driver; torchvision resolves imports.
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.10.0' 'torchvision==0.25.0'
python -m pip install 'transformers>=4.51.0' accelerate bitsandbytes datasets \
  huggingface_hub pandas numpy openpyxl matplotlib tqdm sentencepiece protobuf

if [[ -n "${HF_TOKEN:-}" ]]; then
  printf '%s' "$HF_TOKEN" > "$HF_TOKEN_PATH"
  chmod 600 "$HF_TOKEN_PATH"
  echo 'Hugging Face token stored privately in course cache.'
elif [[ -s "$HF_TOKEN_PATH" ]]; then
  echo 'Existing HF token found.'
elif [[ -t 0 ]]; then
  read -r -s -p 'HF token (input hidden, Enter to skip): ' entered_token
  echo
  if [[ -n "$entered_token" ]]; then
    printf '%s' "$entered_token" > "$HF_TOKEN_PATH"
    chmod 600 "$HF_TOKEN_PATH"
    unset entered_token
  fi
fi
python - <<'PY'
import torch, torchvision, transformers, datasets
print('torch:',torch.__version__,'CUDA build:',torch.version.cuda)
print('torchvision:',torchvision.__version__)
print('transformers:',transformers.__version__)
print('datasets:',datasets.__version__)
PY
printf 'Environment ready: %s\nHugging Face cache: %s\nProject code/results: %s\n' "$VENV_DIR" "$HF_CACHE_DIR" "$PROJECT_DIR"
