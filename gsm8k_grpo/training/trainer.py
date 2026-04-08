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
import signal
import sys
from pathlib import Path
from typing import Optional

import torch

from gsm8k_grpo.config.paths import configure_runtime_environment
from gsm8k_grpo.config.project import TrainingConfig
from gsm8k_grpo.training.model import load_model_and_tokenizer
from gsm8k_grpo.training.runtime_compat import prepare_trl_runtime, require_vllm

_OPTIONAL_BACKEND_ISSUES = prepare_trl_runtime()

logger = logging.getLogger("gsm8k_grpo.trainer")
for _issue in _OPTIONAL_BACKEND_ISSUES:
    logger.warning(_issue)


try:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from gsm8k_grpo.rewards.trl import (
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


def _check_gpu_memory() -> dict:
    """Check available GPU memory and return diagnostics."""
    if not torch.cuda.is_available():
        return {"available": False, "error": "CUDA not available"}
    
    try:
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        free = total - reserved
        
        return {
            "available": True,
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _handle_oom_error(cfg: TrainingConfig, attempt: int, max_attempts: int = 3) -> Optional[TrainingConfig]:
    """Handle OOM errors by reducing batch sizes."""
    if attempt >= max_attempts:
        logger.error("Max OOM retry attempts reached. Giving up.")
        return None
    
    logger.warning(f"OOM error on attempt {attempt + 1}/{max_attempts}")
    
    new_batch_size = max(1, cfg.per_device_train_batch_size // 2)
    new_grad_accum = cfg.gradient_accumulation_steps * 2
    
    if new_batch_size == cfg.per_device_train_batch_size:
        logger.warning("Cannot reduce batch size further. Trying gradient checkpointing.")
    
    logger.info(f"Retrying with batch_size={new_batch_size}, grad_accum={new_grad_accum}")
    
    return cfg._replace(
        per_device_train_batch_size=new_batch_size,
        gradient_accumulation_steps=new_grad_accum,
    )


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

    Reward functions are the four TRL-compatible wrappers from
    gsm8k_grpo.rewards.trl:
      exact_match (weight 1.0) + soft_numeric (0.3) + format (0.2) + length (0.1)

    TRL sums the per-function rewards per completion; the relative weights
    are encoded via RewardWeights in gsm8k_grpo.rewards.

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

class TrainingError(Exception):
    """Custom exception for training failures."""
    pass


def run_training(cfg: TrainingConfig, max_oom_retries: int = 3) -> None:
    """Load model, load data, train, save.
    
    Args:
        cfg: Training configuration
        max_oom_retries: Maximum number of retries on OOM errors
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

    runtime_env = configure_runtime_environment(
        temp_dir=cfg.temp_dir,
        torch_home=cfg.torch_home,
        hf_home=cfg.hf_home,
    )
    logger.info("  tmpdir    : %s", runtime_env["TMPDIR"])
    logger.info("  torchhome : %s", runtime_env["TORCH_HOME"])
    logger.info("  hf_home   : %s", runtime_env["HF_HOME"])
    logger.info("  vllmcache : %s", runtime_env["VLLM_CACHE_ROOT"])
    
    gpu_info = _check_gpu_memory()
    if gpu_info["available"]:
        logger.info(
            "  GPU memory: %.1fGB total, %.1fGB free, %.1fGB used",
            gpu_info["total_gb"],
            gpu_info["free_gb"],
            gpu_info["allocated_gb"],
        )
    else:
        logger.warning("  GPU memory: %s", gpu_info.get("error", "unavailable"))

    if cfg.use_vllm:
        require_vllm("training")

    train_ds = make_trl_dataset(cfg.dataset_path, split="train")
    eval_ds = make_trl_dataset(cfg.dataset_path, split="test")
    
    model = None
    tokenizer = None
    trainer = None
    
    def signal_handler(signum, frame):
        logger.warning("Received interrupt signal. Saving checkpoint...")
        if trainer is not None:
            try:
                trainer.save_model(str(output_dir))
                logger.info("Emergency checkpoint saved.")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")
        logger.info("Shutdown complete.")
        sys.exit(0)
    
    prev_handler = signal.signal(signal.SIGINT, signal_handler)
    
    try:
        oom_attempt = 0
        
        while oom_attempt <= max_oom_retries:
            try:
                model, tokenizer = load_model_and_tokenizer(cfg.model_name, training=True)
                trainer = build_grpo_trainer(cfg, model, tokenizer, train_ds, eval_ds)
                
                trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
                break
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                error_msg = str(e).lower()
                is_oom = "out of memory" in error_msg or "oom" in error_msg
                
                if is_oom and oom_attempt < max_oom_retries:
                    logger.error(f"OOM error on attempt {oom_attempt + 1}: {e}")
                    
                    if trainer is not None:
                        try:
                            trainer.save_model(str(output_dir))
                            logger.info("Checkpoint saved before OOM retry.")
                        except Exception as save_err:
                            logger.warning(f"Could not save checkpoint: {save_err}")
                    
                    del trainer
                    del model
                    if tokenizer is not None:
                        del tokenizer
                    torch.cuda.empty_cache()
                    
                    new_cfg = _handle_oom_error(cfg, oom_attempt, max_oom_retries)
                    if new_cfg is None:
                        raise TrainingError("OOM recovery failed: max retries reached")
                    
                    cfg = new_cfg
                    oom_attempt += 1
                    logger.info(f"Retrying with reduced batch size (attempt {oom_attempt + 1}/{max_oom_retries + 1})")
                else:
                    raise
        
        if trainer is None:
            raise TrainingError("Training failed: trainer not initialized")
            
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")

