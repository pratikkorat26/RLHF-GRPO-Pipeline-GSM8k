"""
Pass-0 baseline evaluation (inference only, no gradients).

Supports two backends:
  - transformers: compatibility path using model.generate()
  - vllm: fast Linux/HPC path for single-GPU inference

Both backends feed the same scoring and reporting pipeline so output shape and
metric semantics stay stable.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

from gsm8k_grpo.data.dataloader import GRPODataset, GRPOCollator
from gsm8k_grpo.config.paths import configure_runtime_environment
from gsm8k_grpo.config.project import EvalConfig
from gsm8k_grpo.rewards.core import composite_reward, exact_match_reward, format_reward
from gsm8k_grpo.training.model import load_model_and_tokenizer, load_tokenizer
from gsm8k_grpo.training.runtime_compat import require_vllm

logger = logging.getLogger("gsm8k_grpo.evaluator")


@dataclass
class EvalResults:
    model_name: str
    split: str
    num_samples: int
    exact_match_accuracy: float
    mean_composite_reward: float
    mean_format_reward: float
    by_difficulty: dict
    timestamp: str


@dataclass
class _EvalExample:
    prompt: str
    reference_answer: str
    metadata: dict


class _EvalAccumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.exact_match_sum = 0.0
        self.composite_sum = 0.0
        self.format_sum = 0.0
        self.by_difficulty: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"n": 0, "exact_match_sum": 0.0, "reward_sum": 0.0}
        )

    def add(self, completion: str, reference: str, metadata: dict) -> None:
        exact = exact_match_reward(completion, reference)
        composite = composite_reward(completion, reference)
        fmt = format_reward(completion)
        difficulty = metadata.get("difficulty", "unknown")

        self.num_samples += 1
        self.exact_match_sum += exact
        self.composite_sum += composite
        self.format_sum += fmt
        bucket = self.by_difficulty[difficulty]
        bucket["n"] += 1
        bucket["exact_match_sum"] += exact
        bucket["reward_sum"] += composite

    def finalize(self, model_name: str, split: str) -> EvalResults:
        n = max(self.num_samples, 1)
        by_difficulty = {
            difficulty: {
                "n": values["n"],
                "accuracy": values["exact_match_sum"] / max(values["n"], 1),
                "mean_reward": values["reward_sum"] / max(values["n"], 1),
            }
            for difficulty, values in self.by_difficulty.items()
        }
        return EvalResults(
            model_name=model_name,
            split=split,
            num_samples=self.num_samples,
            exact_match_accuracy=self.exact_match_sum / n,
            mean_composite_reward=self.composite_sum / n,
            mean_format_reward=self.format_sum / n,
            by_difficulty=by_difficulty,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def _prepare_examples(cfg: EvalConfig, tokenizer) -> list[_EvalExample]:
    hf_jsonl_path = Path(cfg.dataset_path) / "jsonl" / f"{cfg.split}.jsonl"
    dataset = GRPODataset.from_jsonl(str(hf_jsonl_path))
    records = dataset.records

    if cfg.num_samples is not None:
        records = records[: min(cfg.num_samples, len(records))]
        logger.info("Subsampled to %s examples", len(records))

    collator = GRPOCollator(
        tokenizer=tokenizer,
        max_prompt_length=cfg.max_prompt_length,
    )
    examples = [
        _EvalExample(
            prompt=collator._format_prompt(record["prompt_messages"]),
            reference_answer=record["reference_answer"],
            metadata=record.get("metadata", {}),
        )
        for record in records
    ]
    logger.info("Prepared %s prompts for evaluation", len(examples))
    return examples


def _batched_examples(
    examples: list[_EvalExample], batch_size: int
) -> Iterable[list[_EvalExample]]:
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def _run_vllm_evaluation(
    cfg: EvalConfig,
    examples: list[_EvalExample],
) -> _EvalAccumulator:
    require_vllm("evaluation")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg.model_name,
        language_model_only=True,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
    )

    accumulator = _EvalAccumulator()
    for batch_idx, batch in enumerate(_batched_examples(examples, cfg.batch_size), start=1):
        prompts = [item.prompt for item in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for example, output in zip(batch, outputs):
            completion = output.outputs[0].text if output.outputs else ""
            accumulator.add(completion, example.reference_answer, example.metadata)

        if batch_idx % 10 == 0 or accumulator.num_samples == len(examples):
            logger.info("  [%s/%s] records evaluated", accumulator.num_samples, len(examples))

    return accumulator


def _run_transformers_evaluation(
    cfg: EvalConfig,
    tokenizer,
    examples: list[_EvalExample],
) -> _EvalAccumulator:
    model, _ = load_model_and_tokenizer(cfg.model_name)
    model.eval()

    accumulator = _EvalAccumulator()
    for batch in _batched_examples(examples, cfg.batch_size):
        prompts = [example.prompt for example in batch]
        encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=(cfg.temperature > 0.0),
                temperature=cfg.temperature if cfg.temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion_ids = output_ids[:, input_ids.shape[1] :]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for completion, example in zip(completions, batch):
            accumulator.add(completion, example.reference_answer, example.metadata)

        if accumulator.num_samples % max(cfg.batch_size * 10, 1) == 0:
            logger.info("  [%s/%s] records evaluated", accumulator.num_samples, len(examples))

    return accumulator


def run_evaluation(cfg: EvalConfig) -> EvalResults:
    logger.info("=" * 60)
    logger.info("Pass-0 Evaluation")
    logger.info("  model    : %s", cfg.model_name)
    logger.info("  split    : %s", cfg.split)
    logger.info("  backend  : %s", cfg.eval_backend)
    logger.info("  samples  : %s", cfg.num_samples or "all")
    logger.info("  localpath : %s", Path(cfg.model_name).expanduser())
    logger.info("  gpu_mem  : %s", cfg.gpu_memory_utilization)
    logger.info("=" * 60)

    runtime_env = configure_runtime_environment(
        temp_dir=cfg.temp_dir,
        torch_home=cfg.torch_home,
        hf_home=cfg.hf_home,
    )
    logger.info("  tmpdir   : %s", runtime_env["TMPDIR"])
    logger.info("  torchhome: %s", runtime_env["TORCH_HOME"])
    logger.info("  hf_home  : %s", runtime_env["HF_HOME"])
    logger.info("  vllmcache: %s", runtime_env["VLLM_CACHE_ROOT"])

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    tokenizer = load_tokenizer(cfg.model_name)
    examples = _prepare_examples(cfg, tokenizer)

    if cfg.eval_backend == "vllm":
        accumulator = _run_vllm_evaluation(cfg, examples)
    else:
        accumulator = _run_transformers_evaluation(cfg, tokenizer, examples)

    results = accumulator.finalize(cfg.model_name, cfg.split)
    report_path = Path(cfg.output_dir) / "eval_results.json"
    report_path.write_text(
        json.dumps(asdict(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Exact-match accuracy : %.2f%%", results.exact_match_accuracy * 100.0)
    logger.info("Mean composite reward: %.4f", results.mean_composite_reward)
    logger.info("Report saved to      : %s", report_path)
    return results

