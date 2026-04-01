"""
GRPO training logic.

Responsibility: connect the existing data pipeline and reward functions to
TRL's GRPOTrainer. Three public functions:

    make_trl_dataset()    -- load + reformat artifacts from data/
    build_grpo_trainer()  -- configure GRPOConfig + construct GRPOTrainer
    run_training()        -- end-to-end orchestration
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from training.config import TrainingConfig
from training.model import load_model_and_tokenizer
from training.runtime_compat import prepare_trl_runtime, require_vllm

_OPTIONAL_BACKEND_ISSUES = prepare_trl_runtime()

try:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from trl_grpo import (
        exact_match_reward_func,
        format_reward_func,
        length_penalty_func,
        soft_numeric_reward_func,
    )
    _TRL_AVAILABLE = True
except (ImportError, RuntimeError) as _trl_err:
    _TRL_AVAILABLE = False
    _trl_err_msg = str(_trl_err)


def _require_trl() -> None:
    if not _TRL_AVAILABLE:
        raise ImportError(
            "TRL could not be imported in the current environment.\n"
            "Install a compatible stack for standard TRL training:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install transformers datasets trl accelerate\n"
            f"Original error: {_trl_err_msg}"
        )

logger = logging.getLogger("gsm8k_grpo.trainer")
for _issue in _OPTIONAL_BACKEND_ISSUES:
    logger.warning(_issue)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def make_trl_dataset(dataset_path: str, split: str = "train"):  # -> Dataset
    """Load a processed artifact and reformat it for TRL's GRPOTrainer.

    TRL expects each example to have:
      - ``prompt``           : list[dict]  (chat messages, system + user)
      - ``reference_answer`` : str         (ground-truth numeric answer)

    data/pipeline.py writes ``prompt_messages`` — rename it here so the
    calling code in GRPOTrainer can find the column it expects.

    Args:
        dataset_path: Path to the ``trainer/`` artifact directory
                      (contains ``hf_dataset/`` and ``jsonl/`` sub-dirs).
        split: "train" or "test".

    Returns:
        HuggingFace Dataset with ``prompt`` and ``reference_answer`` columns.
    """
    _require_trl()
    hf_path = Path(dataset_path) / "hf_dataset"
    if not hf_path.exists():
        raise FileNotFoundError(
            f"HuggingFace dataset not found at {hf_path}. "
            "Run data/pipeline.py first to generate artifacts."
        )

    from datasets import load_from_disk
    dd = load_from_disk(str(hf_path))
    ds: Dataset = dd[split]

    # Rename prompt_messages → prompt (TRL convention)
    if "prompt_messages" in ds.column_names:
        ds = ds.rename_column("prompt_messages", "prompt")

    # Keep only the columns GRPOTrainer needs
    keep = {"prompt", "reference_answer"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    logger.info(f"TRL dataset ready — split={split}  n={len(ds):,}")
    return ds


# ---------------------------------------------------------------------------
# Trainer construction
# ---------------------------------------------------------------------------

def build_grpo_trainer(
    cfg: TrainingConfig,
    model,  # AutoModelForCausalLM
    tokenizer,  # AutoTokenizer
    train_dataset,  # Dataset
    eval_dataset=None,  # Dataset | None
):  # -> GRPOTrainer
    """Build a TRL GRPOTrainer from a TrainingConfig.

    Reward functions are the four TRL-compatible wrappers from trl_grpo.py:
      exact_match (weight 1.0) + soft_numeric (0.3) + format (0.2) + length (0.1)

    TRL sums the per-function rewards per completion; the relative weights
    are encoded via RewardWeights in trl_grpo.py / reward.py.

    Args:
        cfg: Frozen TrainingConfig.
        model: Policy model (reference model is handled internally by TRL).
        tokenizer: Associated tokenizer.
        train_dataset: TRL-formatted dataset from make_trl_dataset().
        eval_dataset: Optional evaluation split.

    Returns:
        Configured GRPOTrainer, ready to call .train() on.
    """
    _require_trl()
    # Older/newer transformers model classes differ here; TRL expects the dict.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        # --- GRPO algorithm ---
        num_generations=cfg.num_generations,
        beta=cfg.beta,
        use_vllm=cfg.use_vllm,
        # --- batch & optimiser ---
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        # --- sequence lengths ---
        max_completion_length=cfg.max_completion_length,
        # --- hardware ---
        bf16=cfg.bf16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        # --- logging ---
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_cfg,
        reward_funcs=[
            exact_match_reward_func,
            soft_numeric_reward_func,
            format_reward_func,
            length_penalty_func,
        ],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("GRPOTrainer built and ready.")
    return trainer


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------

def run_training(cfg: TrainingConfig) -> None:
    """Load model, load data, train, save.

    This is the single function called by the train_grpo.py entry point.
    """
    logger.info("=" * 60)
    logger.info("GRPO Training Run")
    logger.info(f"  model     : {cfg.model_name}")
    logger.info(f"  data      : {cfg.dataset_path}")
    logger.info(f"  output    : {cfg.output_dir}")
    logger.info(f"  use_vllm  : {cfg.use_vllm}")
    logger.info(f"  G (gens)  : {cfg.num_generations}")
    logger.info(f"  beta (KL) : {cfg.beta}")
    logger.info("=" * 60)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.temp_dir:
        temp_dir = Path(cfg.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(temp_dir)
        os.environ["TMP"] = str(temp_dir)
        os.environ["TEMP"] = str(temp_dir)

    if cfg.use_vllm:
        require_vllm()

    model, tokenizer = load_model_and_tokenizer(cfg.model_name, training=True)
    train_ds = make_trl_dataset(cfg.dataset_path, split="train")
    eval_ds = make_trl_dataset(cfg.dataset_path, split="test")

    trainer = build_grpo_trainer(cfg, model, tokenizer, train_ds, eval_ds)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")
