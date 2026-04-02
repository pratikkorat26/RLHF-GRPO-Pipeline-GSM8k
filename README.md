# RLHF GRPO Pipeline - GSM8K

Production-style data preparation, reward modeling, training, and evaluation pipeline for math reasoning with Group Relative Policy Optimization (GRPO) on [GSM8K](https://huggingface.co/datasets/openai/gsm8k).

---

## Architecture

```text
GSM8K (HuggingFace Hub)
        |
        v
gsm8k_grpo.data.pipeline
  parse answers, build prompts, estimate difficulty
  validate, check leakage, enforce quality gates
        |
        v
trainer/ + analysis/ + reports/
        |
        v
gsm8k_grpo.data.dataloader
  GRPODataset + GRPOCollator -> DataLoader
        |
        v
gsm8k_grpo.rewards.core
  exact_match + soft_numeric + format + length
  composite_reward -> compute_grpo_advantages
```

## Modules

| File | Purpose |
|------|---------|
| `gsm8k_grpo/config/project.py` | Canonical storage, pipeline, training, evaluation, and reward config |
| `gsm8k_grpo/data/pipeline.py` | GSM8K -> GRPO dataset preparation and artifact writing |
| `gsm8k_grpo/data/dataloader.py` | Dataset and collator helpers for training/evaluation |
| `gsm8k_grpo/training/trainer.py` | GRPO training orchestration |
| `gsm8k_grpo/evaluation/evaluator.py` | Pass-0 evaluation and backend selection |
| `gsm8k_grpo/rewards/core.py` | Reward functions and GRPO advantage computation |

## Configuration

Runtime defaults are centralized in [gsm8k_grpo/config/project.py](/E:/learning/SeriousProject/transformers/gsm8k_grpo/config/project.py). That module is the single source of truth for:

- storage/runtime paths
- data pipeline defaults
- training defaults
- evaluation defaults
- reward defaults

Environment variables are used only for storage/runtime concerns such as `PROJECT_STORAGE_ROOT`. Model, data, training, and evaluation defaults live in repo code and can be overridden by CLI flags.

---

## Environment Setup

Use the repo-owned setup scripts once per shell before running the CLI.

WSL/Linux:

```bash
source scripts/setup_env.sh
```

PowerShell:

```powershell
. .\scripts\setup_env.ps1
```

Override the storage root first if you want a different location:

```bash
export PROJECT_STORAGE_ROOT=/custom/path
source scripts/setup_env.sh
```

```powershell
$env:PROJECT_STORAGE_ROOT = "E:\custom\path"
. .\scripts\setup_env.ps1
```

### What `setup_env.sh` does

`setup_env.sh` splits local state into two roots:

- `PROJECT_STORAGE_ROOT`: artifacts and outputs you want to keep
- `PROJECT_RUNTIME_ROOT`: caches and temp files

The script exports:

- `PROJECT_STORAGE_ROOT`
- `PROJECT_RUNTIME_ROOT`
- `TORCH_HOME`
- `HF_HOME`
- `HF_DATASETS_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `VLLM_CACHE_ROOT`
- `TRITON_CACHE_DIR`
- `XDG_CACHE_HOME`
- `TMPDIR`, `TMP`, `TEMP`

It also creates:

- `data/grpo`
- `models/grpo`
- `models/eval`
- `venvs`
- runtime cache directories
- `tmp`

### WSL runtime split

If `PROJECT_STORAGE_ROOT` is under `/mnt/...`, `setup_env.sh` automatically keeps artifacts on that mounted path but moves runtime-only files to a Linux-native path.

Example:

```bash
export PROJECT_STORAGE_ROOT=/mnt/e/learning/SeriousProject/transformers/.localdata
source scripts/setup_env.sh
```

This resolves to:

```bash
PROJECT_STORAGE_ROOT=/mnt/e/learning/SeriousProject/transformers/.localdata
PROJECT_RUNTIME_ROOT=$HOME/.cache/gsm8k-grpo
TORCH_HOME=$HOME/.cache/gsm8k-grpo/torch
HF_HOME=$HOME/.cache/gsm8k-grpo/huggingface
HF_DATASETS_CACHE=$HOME/.cache/gsm8k-grpo/huggingface/datasets
HUGGINGFACE_HUB_CACHE=$HOME/.cache/gsm8k-grpo/huggingface/hub
VLLM_CACHE_ROOT=$HOME/.cache/gsm8k-grpo/vllm
TRITON_CACHE_DIR=$HOME/.cache/gsm8k-grpo/triton
TMPDIR=$HOME/.cache/gsm8k-grpo/tmp
```

That split is required for WSL because `vllm` IPC sockets and temp files do not work reliably on the Windows-mounted `/mnt/...` filesystem.

### Cleanup

Preview the generated local paths that would be removed:

```bash
bash scripts/clean_env.sh
```

Delete them:

```bash
bash scripts/clean_env.sh --yes
```

### What `clean_env.sh` removes

`clean_env.sh` uses the same root-resolution logic as `setup_env.sh`.

It removes generated artifact directories under `PROJECT_STORAGE_ROOT`:

- `data/grpo`
- `models/grpo`
- `models/eval`
- `venvs`

It also removes generated runtime directories under `PROJECT_RUNTIME_ROOT`:

- `torch`
- `huggingface`
- `vllm`
- `triton`
- `tmp`

Current local-root behavior:

- if `PROJECT_STORAGE_ROOT` is repo-local `.localdata`, the whole `.localdata` tree is removed
- if `PROJECT_RUNTIME_ROOT` is `$HOME/.cache/gsm8k-grpo`, that whole runtime tree can be removed too

`bash scripts/clean_env.sh` is a dry-run preview. Nothing is deleted until you pass `--yes`.

### WSL Troubleshooting

If `vllm` fails with IPC or temp-path errors, confirm that:

- `PROJECT_STORAGE_ROOT` is under `/mnt/...`
- `PROJECT_RUNTIME_ROOT` is under `$HOME/.cache/gsm8k-grpo`
- `TMPDIR` is not on `/mnt/...`

Quick check:

```bash
echo "$PROJECT_STORAGE_ROOT"
echo "$PROJECT_RUNTIME_ROOT"
echo "$TMPDIR"
```

If WSL reports `syntax error: unexpected end of file` for the shell scripts, convert them to LF line endings:

```bash
sed -i 's/\r$//' scripts/setup_env.sh scripts/clean_env.sh
chmod +x scripts/setup_env.sh scripts/clean_env.sh
```

---

## Quick Start

### Local WSL/Linux

```bash
export PROJECT_STORAGE_ROOT=/mnt/e/learning/SeriousProject/transformers/.localdata
source scripts/setup_env.sh

echo "$PROJECT_STORAGE_ROOT"
echo "$PROJECT_RUNTIME_ROOT"
echo "$TMPDIR"

python3.11 -m venv .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

python -m gsm8k_grpo.cli.pipeline --splits train test
python -m pytest tests/ -v
```

Smoke evaluation for a few samples:

```bash
python -m gsm8k_grpo.cli.evaluate \
  --model_name Qwen/Qwen3.5-0.8B \
  --eval_backend transformers \
  --num_samples 8 \
  --batch_size 8 \
  --output_dir "$PROJECT_STORAGE_ROOT/models/eval_smoke"
```

### Local PowerShell

```powershell
. .\scripts\setup_env.ps1
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m gsm8k_grpo.cli.pipeline --splits train test
```

### Canonical CLI entrypoints

```bash
python -m gsm8k_grpo.cli.pipeline
python -m gsm8k_grpo.cli.train
python -m gsm8k_grpo.cli.evaluate
```

---

## Storage Defaults

Repo code defaults to this Linux/HPC storage root when `PROJECT_STORAGE_ROOT` is not set:

```bash
/data/cmpe258-sp24/pratikkorat
```

Derived runtime paths:

```bash
TORCH_HOME=/data/cmpe258-sp24/pratikkorat/.cache/torch
HF_HOME=/data/cmpe258-sp24/pratikkorat/.cache/huggingface
HF_DATASETS_CACHE=/data/cmpe258-sp24/pratikkorat/.cache/huggingface/datasets
HUGGINGFACE_HUB_CACHE=/data/cmpe258-sp24/pratikkorat/.cache/huggingface/hub
VLLM_CACHE_ROOT=/data/cmpe258-sp24/pratikkorat/.cache/vllm
TRITON_CACHE_DIR=/data/cmpe258-sp24/pratikkorat/.cache/triton
TMPDIR=/data/cmpe258-sp24/pratikkorat/tmp
dataset artifacts=/data/cmpe258-sp24/pratikkorat/data/grpo
training outputs=/data/cmpe258-sp24/pratikkorat/models/grpo
evaluation outputs=/data/cmpe258-sp24/pratikkorat/models/eval
venv=/data/cmpe258-sp24/pratikkorat/venvs/gsm8k-grpo
```

For WSL local development with artifacts under `/mnt/e/...`, the setup script resolves a split layout like:

```bash
PROJECT_STORAGE_ROOT=/mnt/e/learning/SeriousProject/transformers/.localdata
PROJECT_RUNTIME_ROOT=$HOME/.cache/gsm8k-grpo
TORCH_HOME=$HOME/.cache/gsm8k-grpo/torch
HF_HOME=$HOME/.cache/gsm8k-grpo/huggingface
TMPDIR=$HOME/.cache/gsm8k-grpo/tmp
```

---

## Python API

```python
from gsm8k_grpo.data.pipeline import build_pipeline

trainer_dd = build_pipeline(
    splits=["train", "test"],
    output_dir="./data/grpo",
)
```

## Output Artifacts

```text
data/grpo/
|-- trainer/
|   |-- jsonl/          train.jsonl, test.jsonl
|   `-- hf_dataset/     HuggingFace DatasetDict
|-- analysis/
|   |-- jsonl/          includes reference_solution
|   `-- hf_dataset/
`-- reports/
    |-- manifest.json
    |-- validation_summary.json
    `-- leakage_report.json
```

### Sample trainer record

```json
{
  "idx": 0,
  "split": "train",
  "question": "Natalia sold clips to 48 of her friends...",
  "reference_answer": "72",
  "prompt_messages": [
    {"role": "system", "content": "You are a helpful math tutor..."},
    {"role": "user", "content": "Natalia sold clips..."}
  ],
  "metadata": {"original_idx": 0, "difficulty": "easy"}
}
```

---

## Reward Functions

```python
from gsm8k_grpo.rewards import composite_reward, compute_grpo_advantages

score = composite_reward(completion="Step 1...\n#### 72", reference="72")
advantages = compute_grpo_advantages([0.9, 0.3, 0.7, 1.0, 0.1])
```

| Component | Weight | Measures |
|-----------|--------|----------|
| `exact_match_reward` | 1.0 | Correct final numeric answer |
| `soft_numeric_reward` | 0.3 | Closeness |
| `format_reward` | 0.2 | Presence of `####` plus reasoning structure |
| `length_penalty` | 0.1 | Avoids too-short or too-long responses |

---

## Training and Evaluation

### Linux HPC runbook

```bash
export PROJECT_STORAGE_ROOT=/data/cmpe258-sp24/pratikkorat
source scripts/setup_env.sh

python3.11 -m venv /data/cmpe258-sp24/pratikkorat/venvs/gsm8k-grpo
source /data/cmpe258-sp24/pratikkorat/venvs/gsm8k-grpo/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

python -m gsm8k_grpo.cli.pipeline

python -m gsm8k_grpo.cli.train --model_name Qwen/Qwen3.5-0.8B-Base

python -m gsm8k_grpo.cli.train \
  --model_name Qwen/Qwen3.5-0.8B-Base \
  --resume_from_checkpoint /data/cmpe258-sp24/pratikkorat/models/grpo/checkpoint-500

python -m gsm8k_grpo.cli.evaluate \
  --model_name /data/cmpe258-sp24/pratikkorat/models/grpo/checkpoint-500 \
  --eval_backend vllm \
  --batch_size 64 \
  --gpu_memory_utilization 0.8 \
  --output_dir /data/cmpe258-sp24/pratikkorat/models/eval_full
```

Training defaults to `report_to=none`, so no tracker login is required. Training defaults to `vllm` on the supported Linux/HPC path. Evaluation also defaults to `vllm`, and `--gpu_memory_utilization` defaults to `0.8`.

For local evaluation, `--model_name` must point to either:

- a valid local saved model/checkpoint directory containing tokenizer and model files
- a Hugging Face model id

### Qwen3.5 on HPC

If `vllm` is not aligned with your cluster environment, use the `transformers` backend first:

```bash
python -m gsm8k_grpo.cli.evaluate \
  --model_name Qwen/Qwen3.5-0.8B \
  --eval_backend transformers \
  --num_samples 8 \
  --batch_size 8 \
  --output_dir /data/cmpe258-sp24/pratikkorat/models/eval_qwen_transformers_smoke
```

Once the `vllm` environment is healthy:

```bash
python -m gsm8k_grpo.cli.evaluate \
  --model_name Qwen/Qwen3.5-0.8B \
  --eval_backend vllm \
  --batch_size 64 \
  --gpu_memory_utilization 0.8 \
  --output_dir /data/cmpe258-sp24/pratikkorat/models/eval_qwen_full
```
