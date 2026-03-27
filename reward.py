"""
Reward functions and reward-contract metadata for GRPO on GSM8K.
"""

import logging
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Callable

from data.config import RewardConfig
from utils import FRACTION_RE, NUMERIC_RE, normalise_numeric

logger = logging.getLogger("gsm8k_grpo.reward")

REWARD_CONTRACT_VERSION = "v1"
ANSWER_EXTRACTION_POLICY = "prefer_####_then_last_numeric"
NORMALIZATION_POLICY = "currency_commas_percent_fraction_word_numbers"
_ANSWER_TRIGGER = "####"


def reward_contract_dict(config: RewardConfig | None = None) -> dict:
    cfg = config or RewardConfig()
    payload = cfg.to_dict()
    payload["contract_version"] = REWARD_CONTRACT_VERSION
    payload["answer_extraction_policy"] = ANSWER_EXTRACTION_POLICY
    payload["normalization_policy"] = NORMALIZATION_POLICY
    return payload


def extract_final_answer(completion: str) -> str | None:
    if _ANSWER_TRIGGER in completion:
        after = completion.split(_ANSWER_TRIGGER)[-1].strip()
        result = normalise_numeric(after)
        if result is not None:
            return result

    candidates: list[str] = []
    for m in NUMERIC_RE.finditer(completion):
        val = normalise_numeric(m.group())
        if val is not None:
            candidates.append(val)
    for m in FRACTION_RE.finditer(completion):
        val = normalise_numeric(m.group())
        if val is not None:
            candidates.append(val)

    return candidates[-1] if candidates else None


def _to_float(s: str | None) -> float | None:
    try:
        return float(s) if s is not None else None
    except (ValueError, TypeError):
        return None


def exact_match_reward(completion: str, reference: str) -> float:
    pred = extract_final_answer(completion)
    if pred is None:
        return 0.0

    pred_f, ref_f = _to_float(pred), _to_float(reference.replace(",", ""))
    if pred_f is not None and ref_f is not None:
        return 1.0 if math.isclose(pred_f, ref_f, rel_tol=1e-6) else 0.0

    return 1.0 if pred.strip() == reference.strip() else 0.0


def soft_numeric_reward(completion: str, reference: str, k: float = 0.1) -> float:
    pred = extract_final_answer(completion)
    pred_f, ref_f = _to_float(pred), _to_float(reference.replace(",", ""))
    if pred_f is None or ref_f is None:
        return 0.0

    relative_error = abs(pred_f - ref_f) / (abs(ref_f) + 1.0)
    return math.exp(-k * relative_error)


def format_reward(completion: str) -> float:
    score = 0.0
    has_trigger = _ANSWER_TRIGGER in completion
    if has_trigger:
        score += 0.4
    lines = [l for l in completion.strip().splitlines() if l.strip()]
    if len(lines) >= 3:
        score += 0.3
    if has_trigger:
        after = completion.split(_ANSWER_TRIGGER)[-1]
        if NUMERIC_RE.search(after) or FRACTION_RE.search(after):
            score += 0.3
    return score


def length_penalty(
    completion: str,
    min_tokens: int = 20,  # matches RewardConfig.min_completion_tokens
    max_tokens: int = 512,  # matches RewardConfig.max_completion_tokens
) -> float:
    n = len(completion.split())
    if n < min_tokens:
        return n / min_tokens
    if n > max_tokens:
        return max_tokens / n
    return 1.0


@dataclass(frozen=True)
class RewardWeights:
    exact_match: float = 1.0
    soft_numeric: float = 0.3
    format: float = 0.2
    length: float = 0.1

    @classmethod
    def from_config(cls, config: RewardConfig) -> "RewardWeights":
        return cls(
            exact_match=config.exact_match_weight,
            soft_numeric=config.soft_numeric_weight,
            format=config.format_weight,
            length=config.length_weight,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def composite_reward(
    completion: str,
    reference: str,
    weights: RewardWeights | None = None,
    soft_k: float = 0.1,
) -> float:
    w = weights or RewardWeights()
    r = (
        w.exact_match * exact_match_reward(completion, reference)
        + w.soft_numeric * soft_numeric_reward(completion, reference, k=soft_k)
        + w.format * format_reward(completion)
        + w.length * length_penalty(completion)
    )
    total_weight = w.exact_match + w.soft_numeric + w.format + w.length
    return min(r / total_weight, 1.0)


def compute_group_rewards(
    completions: list[str],
    reference: str,
    reward_fn: Callable[[str, str], float] | None = None,
    weights: RewardWeights | None = None,
    soft_k: float = 0.1,
) -> list[float]:
    fn = reward_fn or (
        lambda c, r: composite_reward(c, r, weights=weights, soft_k=soft_k)
    )
    return [fn(c, reference) for c in completions]


def compute_grpo_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    if len(rewards) == 0:
        return []
    mean_r = statistics.mean(rewards)
    std_r = statistics.pstdev(rewards) + eps
    return [(r - mean_r) / std_r for r in rewards]


def batch_grpo_step(
    prompts_and_completions: list[tuple[str, list[str], str]],
    reward_fn: Callable[[str, str], float] | None = None,
    weights: RewardWeights | None = None,
    soft_k: float = 0.1,
) -> list[dict]:
    results = []
    for prompt, completions, reference in prompts_and_completions:
        rewards = compute_group_rewards(
            completions,
            reference,
            reward_fn=reward_fn,
            weights=weights,
            soft_k=soft_k,
        )
        advantages = compute_grpo_advantages(rewards)
        results.append(
            {
                "prompt": prompt,
                "completions": completions,
                "rewards": rewards,
                "advantages": advantages,
                "reference": reference,
            }
        )
    return results
