"""Package CLI entrypoint for pass-0 evaluation."""

import argparse
import logging
from dataclasses import replace

from gsm8k_grpo.config.project import EvalConfig, ProjectConfig
from gsm8k_grpo.evaluation.evaluator import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def parse_args() -> EvalConfig:
    project_defaults = ProjectConfig()
    defaults = project_defaults.resolved_evaluation()
    p = argparse.ArgumentParser(description="Pass-0 evaluation on GSM8K")

    p.add_argument("--model_name", default=defaults.model_name,
                   help="HuggingFace model ID or local path")
    p.add_argument("--dataset_path", default=defaults.dataset_path,
                   help="Path to data/grpo/trainer artifact directory")
    p.add_argument("--split", default=defaults.split, choices=["train", "test"])
    p.add_argument("--output_dir", default=defaults.output_dir)
    p.add_argument("--temp_dir", default=defaults.temp_dir)
    p.add_argument("--torch_home", default=defaults.torch_home)
    p.add_argument("--hf_home", default=defaults.hf_home)
    p.add_argument("--eval_backend", default=defaults.eval_backend,
                   choices=["vllm", "transformers"],
                   help="Inference backend for evaluation")
    p.add_argument("--num_samples", type=int, default=defaults.num_samples,
                   help="Number of examples to evaluate (default: all)")
    p.add_argument("--batch_size", type=int, default=defaults.batch_size)
    p.add_argument("--max_prompt_length", type=int, default=defaults.max_prompt_length)
    p.add_argument("--max_new_tokens", type=int, default=defaults.max_new_tokens)
    p.add_argument("--temperature", type=float, default=defaults.temperature)
    p.add_argument("--gpu_memory_utilization", type=float, default=defaults.gpu_memory_utilization,
                   help="vLLM GPU memory fraction to reserve")
    p.add_argument("--num_workers", type=int, default=defaults.num_workers,
                   help="DataLoader worker processes for transformers backend")
    p.add_argument("--seed", type=int, default=defaults.seed)

    args = p.parse_args()
    return replace(
        defaults,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        split=args.split,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        torch_home=args.torch_home,
        hf_home=args.hf_home,
        eval_backend=args.eval_backend,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    results = run_evaluation(cfg)

    print(f"\n{'=' * 50}")
    print("Pass-0 Evaluation Results")
    print(f"  Model    : {results.model_name}")
    print(f"  Split    : {results.split}  (n={results.num_samples})")
    print(f"  Accuracy : {results.exact_match_accuracy:.2%}")
    print(f"  Composite: {results.mean_composite_reward:.4f}")
    print(f"  Format   : {results.mean_format_reward:.4f}")
    print("\n  By difficulty:")
    for diff, stats in sorted(results.by_difficulty.items()):
        print(f"    {diff:8s}: acc={stats['accuracy']:.2%}  reward={stats['mean_reward']:.4f}  n={stats['n']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
