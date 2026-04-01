"""
Training and evaluation configuration dataclasses.

All hyperparameters live here — nothing else. Import these into
trainer.py and evaluator.py; override defaults via the CLI entry points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from project_paths import (
    default_eval_output_dir,
    default_hf_home,
    default_temp_dir,
    default_torch_home,
    default_training_dataset_path,
    default_training_output_dir,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Full configuration for a GRPO training run."""

    # --- model & data ---
    model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    dataset_path: str = field(default_factory=default_training_dataset_path)
    output_dir: str = field(default_factory=default_training_output_dir)
    temp_dir: str = field(default_factory=default_temp_dir)
    torch_home: str = field(default_factory=default_torch_home)
    hf_home: str = field(default_factory=default_hf_home)

    # --- GRPO algorithm ---
    num_generations: int = 4    # G: completions sampled per prompt per step
    beta: float = 0.04          # KL-penalty coefficient (divergence from reference)
    use_vllm: bool = True

    # --- batching & optimiser ---
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    max_steps: int = -1          # -1 = run full epochs; positive overrides epochs
    warmup_ratio: float = 0.1

    # --- sequence lengths ---
    max_prompt_length: int = 512
    max_completion_length: int = 512

    # --- hardware ---
    bf16: bool = True
    dataloader_num_workers: int = 0
    seed: int = 42

    # --- logging & checkpointing ---
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int | None = 2
    report_to: str = "none"
    resume_from_checkpoint: str | None = None


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for a pass-0 (inference-only) evaluation run."""

    model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    dataset_path: str = field(default_factory=default_training_dataset_path)
    split: str = "test"
    output_dir: str = field(default_factory=default_eval_output_dir)
    temp_dir: str = field(default_factory=default_temp_dir)
    torch_home: str = field(default_factory=default_torch_home)
    hf_home: str = field(default_factory=default_hf_home)

    eval_backend: str = "vllm"
    num_samples: int | None = None  # None = full split
    batch_size: int = 32
    max_prompt_length: int = 512
    max_new_tokens: int = 512
    temperature: float = 0.0        # 0.0 = greedy decoding
    num_workers: int = 4            # dataloader workers; >0 overlaps tokenization with GPU
    seed: int = 42
