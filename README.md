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
pip install -r requirements.txt

# Build JSONL + HuggingFace Dataset artifacts
python -m data.pipeline --splits train test --output_dir ./data/grpo

# Or with explicit PYTHONPATH
set PYTHONPATH=. && python data/pipeline.py --splits train test --output_dir ./data/grpo

# Run tests
python -m pytest tests/ -v
```

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
