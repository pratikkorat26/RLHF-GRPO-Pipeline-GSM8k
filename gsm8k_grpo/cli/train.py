"""Package CLI entrypoint for GRPO training."""

import argparse
import logging
from dataclasses import replace

from gsm8k_grpo.config.project import ProjectConfig
from gsm8k_grpo.training.config import TrainingConfig
from gsm8k_grpo.training.trainer import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def parse_args() -> TrainingConfig:
    project_defaults = ProjectConfig()
    defaults = project_defaults.resolved_training()
    p = argparse.ArgumentParser(description="GRPO training on GSM8K")

    p.add_argument("--model_name", default=defaults.model_name)
    p.add_argument("--dataset_path", default=defaults.dataset_path)
    p.add_argument("--output_dir", default=defaults.output_dir)
    p.add_argument("--temp_dir", default=defaults.temp_dir)
    p.add_argument("--torch_home", default=defaults.torch_home)
    p.add_argument("--hf_home", default=defaults.hf_home)

    p.add_argument("--num_generations", type=int, default=defaults.num_generations,
                   help="Completions sampled per prompt per step (G)")
    p.add_argument("--beta", type=float, default=defaults.beta,
                   help="KL penalty coefficient")
    p.add_argument("--use_vllm", dest="use_vllm", action="store_true", default=defaults.use_vllm,
                   help="Use vLLM-backed generation during GRPO training")
    p.add_argument("--no_use_vllm", dest="use_vllm", action="store_false",
                   help="Disable vLLM and fall back to standard TRL generation")

    p.add_argument("--batch_size", type=int, default=defaults.per_device_train_batch_size)
    p.add_argument("--grad_accum", type=int, default=defaults.gradient_accumulation_steps)
    p.add_argument("--lr", type=float, default=defaults.learning_rate)
    p.add_argument("--epochs", type=int, default=defaults.num_train_epochs)
    p.add_argument("--max_steps", type=int, default=defaults.max_steps,
                   help="Override epochs; -1 = disabled")
    p.add_argument("--save_steps", type=int, default=defaults.save_steps)
    p.add_argument("--save_total_limit", type=int, default=defaults.save_total_limit)

    p.add_argument("--max_prompt_length", type=int, default=defaults.max_prompt_length)
    p.add_argument("--max_completion_length", type=int, default=defaults.max_completion_length)

    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--resume_from_checkpoint", default=defaults.resume_from_checkpoint)
    p.add_argument("--report_to", default=defaults.report_to,
                   choices=["none", "wandb", "tensorboard"])

    args = p.parse_args()
    return replace(
        defaults,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        torch_home=args.torch_home,
        hf_home=args.hf_home,
        num_generations=args.num_generations,
        beta=args.beta,
        use_vllm=args.use_vllm,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        report_to=args.report_to,
    )


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
