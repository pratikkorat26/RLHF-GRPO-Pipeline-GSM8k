# GSM8K GRPO Training Pipeline

Production-style data preparation, reward modeling, training, and evaluation pipeline for math reasoning with Group Relative Policy Optimization (GRPO) on [GSM8K](https://huggingface.co/datasets/openai/gsm8k).

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [CLI Commands](#cli-commands)
- [Storage](#storage)
- [Reward Functions](#reward-functions)
- [Training](#training)
- [Evaluation](#evaluation)
- [Running Tests](#running-tests)

---

## Overview

This project implements a complete GRPO (Group Relative Policy Optimization) training pipeline:

1. **Data Pipeline** - Downloads and processes GSM8K dataset with validation
2. **Reward Functions** - Multiple reward components (exact match, soft numeric, format, length)
3. **Training** - GRPO training with TRL and optional vLLM backend
4. **Evaluation** - Pass-0 baseline evaluation with transformers or vLLM backend

---

## Quick Start

### WSL2 / Linux

```bash
# 1. Setup environment (all data/caches stored in project directory)
source scripts/setup_env.sh

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate trl vllm tqdm

# 4. Build data pipeline
python -m gsm8k_grpo.cli.pipeline --splits train test

# 5. Run evaluation (smoke test)
python -m gsm8k_grpo.cli.evaluate \
  --model_name Qwen/Qwen3.5-0.8B \
  --eval_backend transformers \
  --num_samples 8 \
  --batch_size 8

# 6. Train model
python -m gsm8k_grpo.cli.train --model_name Qwen/Qwen3.5-0.8B-Base
```

### Docker (Recommended for reproducibility)

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.12 python3-pip

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers datasets accelerate trl vllm tqdm

WORKDIR /workspace
COPY . /workspace

ENV PROJECT_STORAGE_ROOT=/workspace/.localdata
RUN bash scripts/setup_env.sh

CMD ["bash"]
```

Build and run:
```bash
docker build -t gsm8k-grpo .
docker run --gpus all -v $(pwd):/workspace gsm8k-grpo
```

---

## Project Structure

```
transformers/
├── gsm8k_grpo/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── pipeline.py      # Data pipeline CLI
│   │   ├── train.py         # Training CLI
│   │   └── evaluate.py      # Evaluation CLI
│   ├── config/
│   │   ├── __init__.py
│   │   ├── project.py       # All config classes
│   │   └── paths.py         # Path helpers
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pipeline.py      # GSM8K processing
│   │   └── dataloader.py    # PyTorch DataLoader
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # GRPO trainer
│   │   ├── model.py          # Model loading
│   │   └── runtime_compat.py # vLLM compatibility
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py      # Pass-0 evaluation
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── core.py           # Reward functions
│   │   └── trl.py            # TRL integration
│   └── common/
│       ├── __init__.py
│       └── normalization.py   # Numeric parsing
├── scripts/
│   ├── setup_env.sh          # Linux/WSL2 environment setup
│   ├── setup_env.ps1         # Windows PowerShell setup
│   └── clean_env.sh          # Cleanup script
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Unit tests
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Linux / WSL2

```bash
source scripts/setup_env.sh
```

### Windows PowerShell

```powershell
. .\scripts\setup_env.ps1
```

### What `setup_env.sh` Does

Sets environment variables and creates directories:

| Variable | Purpose |
|----------|---------|
| `PROJECT_STORAGE_ROOT` | Base data directory (`.localdata/`) |
| `TORCH_HOME` | PyTorch model cache |
| `HF_HOME` | HuggingFace cache |
| `HF_DATASETS_CACHE` | Downloaded datasets |
| `HUGGINGFACE_HUB_CACHE` | Cached model files |
| `VLLM_CACHE_ROOT` | vLLM compiled kernels |
| `TRITON_CACHE_DIR` | Triton JIT cache |
| `TMPDIR` | Temporary files |

Directories created under `.localdata/`:
```
.localdata/
├── data/grpo/          # Training data
├── models/grpo/        # Trained checkpoints
├── models/eval/        # Evaluation outputs
├── venvs/              # Virtual environments
├── torch/              # PyTorch cache
├── huggingface/        # HuggingFace cache
├── vllm/              # vLLM cache
├── triton/            # Triton cache
└── tmp/               # Temp files
```

### Custom Storage Location

```bash
export PROJECT_STORAGE_ROOT=/custom/path
source scripts/setup_env.sh
```

### Cleanup

Preview what will be deleted:
```bash
bash scripts/clean_env.sh
```

Delete:
```bash
bash scripts/clean_env.sh --yes
```

---

## CLI Commands

### Data Pipeline

```bash
# Build train + test splits
python -m gsm8k_grpo.cli.pipeline --splits train test

# Build only test split
python -m gsm8k_grpo.cli.pipeline --splits test

# Custom output directory
python -m gsm8k_grpo.cli.pipeline --splits train test --output_dir ./my_data

# Custom system prompt
python -m gsm8k_grpo.cli.pipeline --splits train test \
  --system_prompt "Solve the math problem step by step. End with #### answer."
```

### Training

```bash
# Basic training
python -m gsm8k_grpo.cli.train --model_name Qwen/Qwen3.5-0.8B-Base

# Custom settings
python -m gsm8k_grpo.cli.train \
  --model_name Qwen/Qwen3.5-0.8B-Base \
  --num_generations 8 \
  --beta 0.02 \
  --batch_size 4 \
  --lr 1e-5 \
  --epochs 3

# Resume from checkpoint
python -m gsm8k_grpo.cli.train \
  --model_name Qwen/Qwen3.5-0.8B-Base \
  --resume_from_checkpoint .localdata/models/grpo/checkpoint-500

# Use transformers backend (no vLLM)
python -m gsm8k_grpo.cli.train \
  --model_name Qwen/Qwen3.5-0.8B-Base \
  --no_use_vllm
```

### Evaluation

```bash
# vLLM backend (fast, GPU)
python -m gsm8k_grpo.cli.evaluate \
  --model_name Qwen/Qwen3.5-0.8B \
  --eval_backend vllm \
  --batch_size 64

# transformers backend (compatible)
python -m gsm8k_grpo.cli.evaluate \
  --model_name Qwen/Qwen3.5-0.8B \
  --batch_size 8 \
  --num_samples 100

# Local checkpoint
python -m gsm8k_grpo.cli.evaluate \
  --model_name .localdata/models/grpo/checkpoint-500 \
  --eval_backend vllm \
  --batch_size 32
```

---

## Storage

Default storage structure (all under project directory):

```
.localdata/
├── data/grpo/
│   ├── trainer/
│   │   ├── jsonl/          train.jsonl, test.jsonl
│   │   └── hf_dataset/     HuggingFace DatasetDict
│   ├── analysis/
│   │   ├── jsonl/          Includes reference_solution
│   │   └── hf_dataset/
│   └── reports/
│       ├── manifest.json
│       ├── validation_summary.json
│       └── leakage_report.json
├── models/grpo/             # Training checkpoints
│   └── checkpoint-*/        # Saved checkpoints
└── models/eval/            # Evaluation results
    └── eval_results.json
```

### Sample Data Record

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

### Components

| Function | Weight | Description |
|----------|--------|-------------|
| `exact_match_reward` | 1.0 | Correct final numeric answer |
| `soft_numeric_reward` | 0.3 | Exponential decay based on relative error |
| `format_reward` | 0.2 | Contains `####`, has reasoning lines, numeric after marker |
| `length_penalty` | 0.1 | Penalizes too short (<20) or too long (>512) responses |

### Python API

```python
from gsm8k_grpo.rewards.core import (
    composite_reward,
    compute_grpo_advantages,
    exact_match_reward,
    format_reward,
    soft_numeric_reward,
)

# Score a completion
score = composite_reward(
    completion="Step 1.\nStep 2.\n#### 72",
    reference="72"
)

# Compute group advantages for GRPO
rewards = [0.9, 0.3, 0.7, 1.0, 0.1]
advantages = compute_grpo_advantages(rewards)
```

### Answer Extraction

The pipeline extracts answers using the `####` marker, falling back to the last numeric value in the completion. Supports:
- Integers: `42`, `-15`
- Decimals: `3.14`, `-0.5`
- Fractions: `1/2` → `0.5`
- Currency: `$1,200` → `1200`
- Percentages: `35%` → `35`
- Comma-separated: `1,024` → `1024`
- Word numbers: `twelve` → `12`

---

## Training

### GRPO Configuration Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen3.5-0.8B-Base` | Base model |
| `num_generations` | 4 | Completions per prompt |
| `beta` | 0.04 | KL penalty coefficient |
| `batch_size` | 2 | Per-device batch size |
| `gradient_accumulation_steps` | 4 | Effective batch = 8 |
| `learning_rate` | 1e-5 | Optimizer LR |
| `epochs` | 1 | Training epochs |
| `max_completion_length` | 512 | Max generation tokens |
| `use_vllm` | true | Use vLLM for generation |

### Training Output

```
.localdata/models/grpo/
├── checkpoint-500/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── trainer_state.json
└── checkpoint-1000/
```

---

## Evaluation

### Evaluation Results Format

```json
{
  "model_name": "Qwen/Qwen3.5-0.8B",
  "split": "test",
  "num_samples": 1319,
  "exact_match_accuracy": 0.42,
  "mean_composite_reward": 0.65,
  "mean_format_reward": 0.71,
  "by_difficulty": {
    "easy": {"n": 500, "accuracy": 0.55, "mean_reward": 0.72},
    "medium": {"n": 500, "accuracy": 0.38, "mean_reward": 0.61},
    "hard": {"n": 319, "accuracy": 0.28, "mean_reward": 0.54}
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_pipeline.py::TestRewardFunctions -v

# Run with coverage
python -m pytest tests/ -v --cov=gsm8k_grpo
```

### Test Categories

- `TestNormaliseNumeric` - Numeric parsing
- `TestParsing` - GSM8K answer extraction
- `TestRecordValidation` - Data validation
- `TestRewardFunctions` - Reward computation
- `TestGRPOAdvantages` - Advantage computation
- `TestLeakageChecks` - Cross-split leakage detection
- `TestBuildPipeline` - End-to-end pipeline

---

## Dependencies

```
transformers
datasets
accelerate
trl
vllm
tqdm
torch (CUDA 12.1)
```

Install:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate trl vllm tqdm
```

---

## Configuration

All defaults are in `gsm8k_grpo/config/project.py`. Override via CLI flags or environment variables:

```python
from gsm8k_grpo.config.project import ProjectConfig, TrainingConfig

# Get resolved config
cfg = ProjectConfig()
training_cfg = cfg.resolved_training()

# Override
training_cfg = training_cfg._replace(
    model_name="Qwen/Qwen3.5-1.5B-Base",
    learning_rate=5e-6,
)
```
