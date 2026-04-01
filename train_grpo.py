"""
Entry point for GRPO training.

All logic lives in training/trainer.py. This script just parses CLI
arguments, builds a TrainingConfig, and calls run_training().

Usage:
    python train_grpo.py
    python train_grpo.py --model_name Qwen/Qwen2.5-0.5B --lr 5e-6
    python train_grpo.py --max_steps 1   # dry-run (1 step)
    accelerate launch train_grpo.py      # multi-GPU
"""

import argparse
import logging

from training.config import TrainingConfig
from training.trainer import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def parse_args() -> TrainingConfig:
    defaults = TrainingConfig()
    p = argparse.ArgumentParser(description="GRPO training on GSM8K")

    # model & data
    p.add_argument("--model_name", default=defaults.model_name)
    p.add_argument("--dataset_path", default=defaults.dataset_path)
    p.add_argument("--output_dir", default=defaults.output_dir)

    # GRPO algorithm
    p.add_argument("--num_generations", type=int, default=defaults.num_generations,
                   help="Completions sampled per prompt per step (G)")
    p.add_argument("--beta", type=float, default=defaults.beta,
                   help="KL penalty coefficient")

    # batching & optimiser
    p.add_argument("--batch_size", type=int, default=defaults.per_device_train_batch_size)
    p.add_argument("--grad_accum", type=int, default=defaults.gradient_accumulation_steps)
    p.add_argument("--lr", type=float, default=defaults.learning_rate)
    p.add_argument("--epochs", type=int, default=defaults.num_train_epochs)
    p.add_argument("--max_steps", type=int, default=defaults.max_steps,
                   help="Override epochs; -1 = disabled")

    # sequence lengths
    p.add_argument("--max_prompt_length", type=int, default=defaults.max_prompt_length)
    p.add_argument("--max_completion_length", type=int, default=defaults.max_completion_length)

    # misc
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--report_to", default=defaults.report_to,
                   choices=["none", "wandb", "tensorboard"])

    args = p.parse_args()
    return TrainingConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        seed=args.seed,
        report_to=args.report_to,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
