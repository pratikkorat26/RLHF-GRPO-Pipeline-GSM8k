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

Default script roots:

- WSL/Linux: `$HOME/gsm8k-grpo`
- PowerShell: `<repo>/.localdata`

The scripts create and export:

- `PROJECT_STORAGE_ROOT`
- `TORCH_HOME`
- `HF_HOME`
- `HF_DATASETS_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `VLLM_CACHE_ROOT`
- `TRITON_CACHE_DIR`
- `XDG_CACHE_HOME`
- `TMPDIR`, `TMP`, `TEMP`

They also create:

- `data/grpo`
- `models/grpo`
- `models/eval`
- `venvs`
- `.cache/...`
- `tmp`

---

## Quick Start

### Local WSL/Linux

```bash
source scripts/setup_env.sh
python3.11 -m venv .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

python -m gsm8k_grpo.cli.pipeline --splits train test
python -m pytest tests/ -v
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
