#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
default_storage_root="${repo_root}/.localdata"

is_wsl() {
  if grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null; then
    return 0
  fi
  if uname -r 2>/dev/null | grep -qiE '(microsoft|wsl)'; then
    return 0
  fi
  return 1
}

PROJECT_STORAGE_ROOT="${PROJECT_STORAGE_ROOT:-$default_storage_root}"
export PROJECT_STORAGE_ROOT

if [[ -z "${PROJECT_RUNTIME_ROOT:-}" ]]; then
  if is_wsl && [[ "$PROJECT_STORAGE_ROOT" == /mnt/* ]]; then
    PROJECT_RUNTIME_ROOT="$HOME/.cache/gsm8k-grpo"
  else
    PROJECT_RUNTIME_ROOT="$PROJECT_STORAGE_ROOT"
  fi
fi
export PROJECT_RUNTIME_ROOT

export TORCH_HOME="${TORCH_HOME:-$PROJECT_RUNTIME_ROOT/torch}"
export HF_HOME="${HF_HOME:-$PROJECT_RUNTIME_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PROJECT_RUNTIME_ROOT/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_RUNTIME_ROOT/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_RUNTIME_ROOT}"
export TMPDIR="${TMPDIR:-$PROJECT_RUNTIME_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"

mkdir -p \
  "$PROJECT_STORAGE_ROOT/data/grpo" \
  "$PROJECT_STORAGE_ROOT/models/grpo" \
  "$PROJECT_STORAGE_ROOT/models/eval" \
  "$PROJECT_STORAGE_ROOT/venvs" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR" \
  "$TMPDIR"

printf 'PROJECT_STORAGE_ROOT=%s\n' "$PROJECT_STORAGE_ROOT"
printf 'PROJECT_RUNTIME_ROOT=%s\n' "$PROJECT_RUNTIME_ROOT"
printf 'TORCH_HOME=%s\n' "$TORCH_HOME"
printf 'HF_HOME=%s\n' "$HF_HOME"
printf 'HF_DATASETS_CACHE=%s\n' "$HF_DATASETS_CACHE"
printf 'HUGGINGFACE_HUB_CACHE=%s\n' "$HUGGINGFACE_HUB_CACHE"
printf 'VLLM_CACHE_ROOT=%s\n' "$VLLM_CACHE_ROOT"
printf 'TRITON_CACHE_DIR=%s\n' "$TRITON_CACHE_DIR"
printf 'XDG_CACHE_HOME=%s\n' "$XDG_CACHE_HOME"
printf 'TMPDIR=%s\n' "$TMPDIR"
