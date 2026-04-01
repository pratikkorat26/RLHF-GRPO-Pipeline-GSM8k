"""
Entry point for pass-0 baseline evaluation.

All logic lives in training/evaluator.py. This script parses CLI
arguments, builds an EvalConfig, runs evaluation, and prints a summary.

Usage:
    python evaluate.py
    python evaluate.py --split test --num_samples 100
    python evaluate.py --model_name ./output/grpo  # evaluate fine-tuned model
"""

import argparse
import logging

from training.config import EvalConfig
from training.evaluator import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def parse_args() -> EvalConfig:
    defaults = EvalConfig()
    p = argparse.ArgumentParser(description="Pass-0 evaluation on GSM8K")

    p.add_argument("--model_name", default=defaults.model_name,
                   help="HuggingFace model ID or local path")
    p.add_argument("--dataset_path", default=defaults.dataset_path,
                   help="Path to data/grpo/trainer artifact directory")
    p.add_argument("--split", default=defaults.split, choices=["train", "test"])
    p.add_argument("--output_dir", default=defaults.output_dir)
    p.add_argument("--num_samples", type=int, default=defaults.num_samples,
                   help="Number of examples to evaluate (default: all)")
    p.add_argument("--batch_size", type=int, default=defaults.batch_size)
    p.add_argument("--max_new_tokens", type=int, default=defaults.max_new_tokens)
    p.add_argument("--num_workers", type=int, default=defaults.num_workers,
                   help="DataLoader worker processes")
    p.add_argument("--seed", type=int, default=defaults.seed)

    args = p.parse_args()
    return EvalConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        split=args.split,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    results = run_evaluation(cfg)

    print(f"\n{'=' * 50}")
    print(f"Pass-0 Evaluation Results")
    print(f"  Model    : {results.model_name}")
    print(f"  Split    : {results.split}  (n={results.num_samples})")
    print(f"  Accuracy : {results.exact_match_accuracy:.2%}")
    print(f"  Composite: {results.mean_composite_reward:.4f}")
    print(f"  Format   : {results.mean_format_reward:.4f}")
    print(f"\n  By difficulty:")
    for diff, stats in sorted(results.by_difficulty.items()):
        print(f"    {diff:8s}: acc={stats['accuracy']:.2%}  reward={stats['mean_reward']:.4f}  n={stats['n']}")
    print(f"{'=' * 50}")
