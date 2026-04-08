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
from typing import Iterable, Optional

import torch

from gsm8k_grpo.data.dataloader import GRPODataset, GRPOCollator
from gsm8k_grpo.config.paths import configure_runtime_environment
from gsm8k_grpo.config.project import EvalConfig
from gsm8k_grpo.rewards.core import composite_reward, exact_match_reward, format_reward
from gsm8k_grpo.training.model import load_model_and_tokenizer, load_tokenizer
from gsm8k_grpo.training.runtime_compat import require_vllm

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger("gsm8k_grpo.evaluator")


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
    total_batches = (len(examples) + cfg.batch_size - 1) // cfg.batch_size

    for batch_idx, batch in enumerate(
        tqdm(
            _batched_examples(examples, cfg.batch_size),
            total=total_batches,
            desc="Evaluating (vLLM)",
        ),
        start=1,
    ):
        prompts = [item.prompt for item in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for example, output in zip(batch, outputs):
            completion = output.outputs[0].text if output.outputs else ""
            accumulator.add(completion, example.reference_answer, example.metadata)

    return accumulator


def _run_transformers_evaluation(
    cfg: EvalConfig,
    tokenizer,
    examples: list[_EvalExample],
) -> _EvalAccumulator:
    model, _ = load_model_and_tokenizer(cfg.model_name)
    model.eval()

    accumulator = _EvalAccumulator()
    total_batches = (len(examples) + cfg.batch_size - 1) // cfg.batch_size

    for batch in tqdm(
        _batched_examples(examples, cfg.batch_size),
        total=total_batches,
        desc="Evaluating (transformers)",
    ):
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

    return accumulator


def run_evaluation(cfg: EvalConfig, max_oom_retries: int = 2) -> EvalResults:
    """Run evaluation with optional OOM retry logic.
    
    Args:
        cfg: Evaluation configuration
        max_oom_retries: Maximum retries on OOM errors
    """
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

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    tokenizer = load_tokenizer(cfg.model_name)
    examples = _prepare_examples(cfg, tokenizer)
    
    accumulator = None
    oom_attempt = 0
    
    while oom_attempt <= max_oom_retries:
        try:
            if cfg.eval_backend == "vllm":
                accumulator = _run_vllm_evaluation(cfg, examples)
            else:
                accumulator = _run_transformers_evaluation(cfg, tokenizer, examples)
            break
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e).lower()
            is_oom = "out of memory" in error_msg or "oom" in error_msg
            
            if is_oom and oom_attempt < max_oom_retries:
                logger.error(f"OOM error on attempt {oom_attempt + 1}: {e}")
                torch.cuda.empty_cache()
                
                new_batch_size = max(1, cfg.batch_size // 2)
                logger.info(f"Retrying with reduced batch_size={new_batch_size}")
                
                cfg = cfg._replace(batch_size=new_batch_size)
                oom_attempt += 1
            else:
                raise
    
    if accumulator is None:
        raise RuntimeError("Evaluation failed: accumulator not initialized")

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

