# RLHF GRPO Pipeline — GSM8K

Production-style data preparation and reward modeling pipeline for training math reasoning models with **Group Relative Policy Optimization (GRPO)** on the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset.

---

## Architecture

```
GSM8K (HuggingFace Hub)
        │  load_gsm8k()
        ▼
  pipeline.py  ──── parse answers, build prompts, estimate difficulty
        │            validate, check leakage, enforce quality gates
        │
   ┌────┴────┐
   ▼         ▼
trainer/   analysis/          data/grpo/reports/
(public    (+ reference_       manifest.json
 fields)    solution)          validation_summary.json
                               leakage_report.json
        │
        ▼
  dataloader.py  ──── GRPODataset + GRPOCollator → DataLoader
        │
        ▼
  reward.py      ──── exact_match · soft_numeric · format · length
                      composite_reward → compute_grpo_advantages
```

---

## Modules

| File | Purpose |
|------|---------|
| `utils.py` | Shared numeric normalization (`normalise_numeric`) |
| `config.py` | Frozen dataclasses: `PipelineConfig`, `RewardConfig` |
| `pipeline.py` | GSM8K → GRPO dataset (parse, validate, save artifacts) |
| `dataloader.py` | `GRPODataset`, `GRPOCollator`, `build_dataloader` |
| `reward.py` | Reward functions + GRPO advantage computation |
| `tests/` | Unit tests (~90% coverage, no network calls) |

---

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

# Build JSONL + HuggingFace Dataset artifacts
python -m data.pipeline --splits train test --output_dir ./data/grpo

# Run tests
python -m pytest tests/ -v
```

The commands above are the supported Linux path. Hugging Face and Torch caches
are expected to be configured externally on the target machine.

By default, the project now assumes the following Linux storage root:

```bash
/data/cmpe258-sp24/pratikkorat
```

Default runtime paths derived from that root:

```bash
TORCH_HOME=/data/cmpe258-sp24/pratikkorat/.cache/torch
HF_HOME=/data/cmpe258-sp24/pratikkorat/.cache/huggingface
HF_DATASETS_CACHE=/data/cmpe258-sp24/pratikkorat/.cache/huggingface/datasets
TRANSFORMERS_CACHE=/data/cmpe258-sp24/pratikkorat/.cache/huggingface/hub
VLLM_CACHE_ROOT=/data/cmpe258-sp24/pratikkorat/.cache/vllm
TRITON_CACHE_DIR=/data/cmpe258-sp24/pratikkorat/.cache/triton
TMPDIR=/data/cmpe258-sp24/pratikkorat/tmp
dataset artifacts=/data/cmpe258-sp24/pratikkorat/data/grpo
training outputs=/data/cmpe258-sp24/pratikkorat/models/grpo
evaluation outputs=/data/cmpe258-sp24/pratikkorat/models/eval
venv=/data/cmpe258-sp24/pratikkorat/venvs/gsm8k-grpo
```

Set `PROJECT_STORAGE_ROOT` only if you need to override that root.

### Python API

```python
from data.pipeline import build_pipeline

trainer_dd = build_pipeline(
    splits=["train", "test"],
    output_dir="./data/grpo",
)
```

---

## Output Artifacts

```
data/grpo/
├── trainer/
│   ├── jsonl/          train.jsonl, test.jsonl  (public fields only)
│   └── hf_dataset/     HuggingFace DatasetDict
├── analysis/
│   ├── jsonl/          includes reference_solution (CoT)
│   └── hf_dataset/
└── reports/
    ├── manifest.json            config + stats snapshot
    ├── validation_summary.json  per-split quality metrics
    └── leakage_report.json      cross-split duplicate check
```

### Sample record (trainer split)

```json
{
  "idx": 0,
  "split": "train",
  "question": "Natalia sold clips to 48 of her friends...",
  "reference_answer": "72",
  "prompt_messages": [
    {"role": "system", "content": "You are a helpful math tutor..."},
    {"role": "user",   "content": "Natalia sold clips..."}
  ],
  "metadata": {"original_idx": 0, "difficulty": "easy"}
}
```

---

## Reward Functions

```python
from reward import composite_reward, compute_grpo_advantages

# Score a single completion
score = composite_reward(completion="Step 1...\n#### 72", reference="72")

# GRPO advantage normalization for a group of completions
rewards    = [0.9, 0.3, 0.7, 1.0, 0.1]
advantages = compute_grpo_advantages(rewards)  # zero-mean, unit-std
```

| Component | Weight | Measures |
|-----------|--------|----------|
| `exact_match_reward` | 1.0 | Correct final numeric answer |
| `soft_numeric_reward` | 0.3 | Closeness (exponential penalty) |
| `format_reward` | 0.2 | Presence of `####` + step-by-step lines |
| `length_penalty` | 0.1 | Avoids too-short or too-long responses |

---

## TRL Integration

This package integrates with HuggingFace's [TRL](https://huggingface.co/docs/trl) library for GRPO training.

### Quick Start

```python
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from trl_grpo import exact_match_reward_func, soft_numeric_reward_func, format_reward_func

dataset = Dataset.from_list([
    {"prompt": "What is 2+2?", "reference_answer": "4"},
    {"prompt": "What is 5*3?", "reference_answer": "15"},
])

training_args = GRPOConfig(
    output_dir="./grpo_output",
    per_device_train_batch_size=4,
    num_generations=4,
    max_completion_length=512,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    args=training_args,
    reward_funcs=[
        exact_match_reward_func,
        soft_numeric_reward_func,
        format_reward_func,
    ],
    train_dataset=dataset,
)

trainer.train()
```

### Using Composite Reward

```python
from trl import GRPOTrainer, GRPOConfig
from trl_grpo import create_reward_func, RewardWeights

custom_weights = RewardWeights(
    exact_match=1.0,
    soft_numeric=0.3,
    format=0.2,
    length=0.1,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=create_reward_func(weights=custom_weights, soft_k=0.1),
    train_dataset=dataset,
)
```

### Linux HPC Runbook

```bash
python3.11 -m venv /data/cmpe258-sp24/pratikkorat/venvs/gsm8k-grpo
source /data/cmpe258-sp24/pratikkorat/venvs/gsm8k-grpo/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

# Build dataset artifacts if needed
python -m data.pipeline

# Single-GPU H100 training; vLLM is enabled by default
python train_grpo.py --model_name Qwen/Qwen3.5-0.8B-Base

# Resume from the latest checkpoint
python train_grpo.py \
  --model_name Qwen/Qwen3.5-0.8B-Base \
  --resume_from_checkpoint /data/cmpe258-sp24/pratikkorat/models/grpo/checkpoint-500

# Optional fallback path without vLLM
python train_grpo.py \
  --model_name Qwen/Qwen3.5-0.8B-Base \
  --output_dir /data/cmpe258-sp24/pratikkorat/models/grpo_no_vllm \
  --no_use_vllm

# Evaluate a trained checkpoint or final export
python evaluate.py \
  --model_name /data/cmpe258-sp24/pratikkorat/models/grpo \
  --eval_backend vllm \
  --batch_size 32 \
  --output_dir /data/cmpe258-sp24/pratikkorat/models/eval

# Evaluate the full test split; omit --num_samples for a full-dataset run
python evaluate.py \
  --model_name /data/cmpe258-sp24/pratikkorat/models/grpo \
  --eval_backend vllm \
  --batch_size 64 \
  --output_dir /data/cmpe258-sp24/pratikkorat/models/eval_full

# Transformers fallback for debugging or backend comparison
python evaluate.py \
  --model_name /data/cmpe258-sp24/pratikkorat/models/grpo \
  --eval_backend transformers \
  --num_samples 8 \
  --batch_size 8 \
  --output_dir /data/cmpe258-sp24/pratikkorat/models/eval_transformers
```

Training defaults to `report_to=none`, so no tracker login is required. GRPO
training also defaults to `vllm` generation on the supported Linux/HPC path.
Evaluation also defaults to the `vllm` backend on Linux/HPC. Leave
`--num_samples` unset to evaluate the full selected split.

If you need a scheduler wrapper, a minimal single-GPU job script looks like:

```bash
#!/bin/bash
#SBATCH --job-name=gsm8k-grpo
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source /data/cmpe258-sp24/pratikkorat/venvs/gsm8k-grpo/bin/activate

python train_grpo.py \
  --model_name Qwen/Qwen3.5-0.8B-Base
```
