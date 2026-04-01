"""
Pass-0 baseline evaluation (inference only, no gradients).

Loads the base model, runs greedy decoding on the test split, scores
each completion with the existing reward functions, and saves a JSON
report. This gives a pre-training baseline before any GRPO fine-tuning.

Public API:
    run_evaluation(cfg: EvalConfig) -> EvalResults
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

import torch
from torch.utils.data import DataLoader

from data.dataloader import GRPODataset, build_dataloader
from reward import composite_reward, exact_match_reward, format_reward
from training.config import EvalConfig
from training.model import load_model_and_tokenizer

logger = logging.getLogger("gsm8k_grpo.evaluator")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    model_name: str
    split: str
    num_samples: int
    exact_match_accuracy: float     # fraction of examples with correct answer
    mean_composite_reward: float    # averaged composite_reward over all samples
    mean_format_reward: float       # averaged format_reward over all samples
    by_difficulty: dict             # {"easy": {"n": int, "accuracy": float, "mean_reward": float}}
    timestamp: str


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg: EvalConfig) -> EvalResults:
    """Run pass-0 evaluation: load model → generate → score → save JSON.

    Args:
        cfg: Frozen EvalConfig.

    Returns:
        EvalResults with accuracy and reward statistics.
    """
    logger.info("=" * 60)
    logger.info("Pass-0 Evaluation")
    logger.info(f"  model    : {cfg.model_name}")
    logger.info(f"  split    : {cfg.split}")
    logger.info(f"  samples  : {cfg.num_samples or 'all'}")
    logger.info("=" * 60)

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # --- 1. Load model & tokenizer ---
    model, tokenizer = load_model_and_tokenizer(cfg.model_name)
    model.eval()

    # --- 2. Load dataset ---
    hf_jsonl_path = Path(cfg.dataset_path) / "jsonl" / f"{cfg.split}.jsonl"
    dataset = GRPODataset.from_jsonl(str(hf_jsonl_path))

    if cfg.num_samples is not None:
        indices = list(range(min(cfg.num_samples, len(dataset))))
        dataset = GRPODataset([dataset[i] for i in indices])
        logger.info(f"Subsampled to {len(dataset)} examples")

    loader: DataLoader = build_dataloader(
        dataset,
        tokenizer,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        max_prompt_length=512,
        pin_memory=torch.cuda.is_available(),
    )

    # --- 3. Generate & score ---
    per_sample: list[dict] = []

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        references: list[str] = batch["reference_answers"]
        metadata_list: list[dict] = batch["metadata"]
        prompt_strings: list[str] = batch["prompt_strings"]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=(cfg.temperature > 0.0),
                temperature=cfg.temperature if cfg.temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Slice off the prompt tokens — keep only the generated part
        completion_ids = output_ids[:, input_ids.shape[1]:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        for completion, reference, meta in zip(completions, references, metadata_list):
            em = exact_match_reward(completion, reference)
            cr = composite_reward(completion, reference)
            fr = format_reward(completion)
            difficulty = meta.get("difficulty", "unknown")

            per_sample.append({
                "exact_match": em,
                "composite_reward": cr,
                "format_reward": fr,
                "difficulty": difficulty,
            })

        if (batch_idx + 1) % 10 == 0:
            done = (batch_idx + 1) * cfg.batch_size
            logger.info(f"  [{done}/{len(dataset)}] batches evaluated")

    # --- 4. Aggregate ---
    n = len(per_sample)
    accuracy = sum(s["exact_match"] for s in per_sample) / max(n, 1)
    mean_composite = sum(s["composite_reward"] for s in per_sample) / max(n, 1)
    mean_format = sum(s["format_reward"] for s in per_sample) / max(n, 1)

    # Per-difficulty breakdown
    by_diff: dict[str, dict] = defaultdict(lambda: {"n": 0, "exact_match_sum": 0.0, "reward_sum": 0.0})
    for s in per_sample:
        d = s["difficulty"]
        by_diff[d]["n"] += 1
        by_diff[d]["exact_match_sum"] += s["exact_match"]
        by_diff[d]["reward_sum"] += s["composite_reward"]

    by_difficulty = {
        d: {
            "n": v["n"],
            "accuracy": v["exact_match_sum"] / max(v["n"], 1),
            "mean_reward": v["reward_sum"] / max(v["n"], 1),
        }
        for d, v in by_diff.items()
    }

    results = EvalResults(
        model_name=cfg.model_name,
        split=cfg.split,
        num_samples=n,
        exact_match_accuracy=accuracy,
        mean_composite_reward=mean_composite,
        mean_format_reward=mean_format,
        by_difficulty=by_difficulty,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # --- 5. Save JSON report ---
    report_path = Path(cfg.output_dir) / "eval_results.json"
    report_path.write_text(
        json.dumps(asdict(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Exact-match accuracy : {accuracy:.2%}")
    logger.info(f"Mean composite reward: {mean_composite:.4f}")
    logger.info(f"Report saved to      : {report_path}")
    return results
