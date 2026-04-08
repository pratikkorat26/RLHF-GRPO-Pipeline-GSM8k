#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
default_storage_root="${repo_root}/.localdata"

dry_run=1
for arg in "$@"; do
  case "$arg" in
    --yes)
      dry_run=0
      ;;
    --dry-run)
      dry_run=1
      ;;
    *)
      printf 'Unknown argument: %s\n' "$arg" >&2
      printf 'Usage: bash scripts/clean_env.sh [--dry-run] [--yes]\n' >&2
      exit 2
      ;;
  esac
done

PROJECT_STORAGE_ROOT="${PROJECT_STORAGE_ROOT:-$default_storage_root}"
PROJECT_RUNTIME_ROOT="$PROJECT_STORAGE_ROOT"

targets=(
  "$PROJECT_STORAGE_ROOT/data/grpo"
  "$PROJECT_STORAGE_ROOT/models/grpo"
  "$PROJECT_STORAGE_ROOT/models/eval"
  "$PROJECT_STORAGE_ROOT/venvs"
  "$PROJECT_RUNTIME_ROOT/torch"
  "$PROJECT_RUNTIME_ROOT/huggingface"
  "$PROJECT_RUNTIME_ROOT/vllm"
  "$PROJECT_RUNTIME_ROOT/triton"
  "$PROJECT_RUNTIME_ROOT/tmp"
)

printf 'PROJECT_STORAGE_ROOT=%s\n' "$PROJECT_STORAGE_ROOT"
printf 'PROJECT_RUNTIME_ROOT=%s\n' "$PROJECT_RUNTIME_ROOT"
printf 'Cleanup mode=%s\n' "$(if [[ "$dry_run" -eq 1 ]]; then printf 'dry-run'; else printf 'delete'; fi)"
printf 'Targets:\n'
for target in "${targets[@]}"; do
  printf '  %s\n' "$target"
done

if [[ "$dry_run" -eq 1 ]]; then
  printf '\nPreview only. Re-run with --yes to delete these paths.\n'
  exit 0
fi

for target in "${targets[@]}"; do
  if [[ -e "$target" ]]; then
    rm -rf -- "$target"
    printf 'Removed %s\n' "$target"
  else
    printf 'Skipped missing %s\n' "$target"
  fi
done
