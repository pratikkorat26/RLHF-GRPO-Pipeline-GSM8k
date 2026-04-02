"""
TRL GRPOTrainer integration for GSM8K GRPO training.

Wraps reward functions from gsm8k_grpo.rewards.core for use with HuggingFace TRL's GRPOTrainer.
"""

from typing import Any

from gsm8k_grpo.rewards.core import (
    RewardWeights,
    composite_reward,
    exact_match_reward,
    format_reward,
    length_penalty,
    soft_numeric_reward,
)


def _to_text(completion) -> str:
    """Normalise a TRL completion to a plain string.

    Older TRL passes completions as str.
    Newer TRL passes them as list[dict] (chat message format):
        [{"role": "assistant", "content": "..."}]
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


def create_reward_func(
    weights: RewardWeights | None = None,
    soft_k: float = 0.1,
    use_composite: bool = True,
    use_exact: bool = True,
    use_soft_numeric: bool = True,
    use_format: bool = True,
    use_length: bool = True,
):
    """Create a TRL-compatible reward function with configurable components.

    Args:
        weights: Reward weights for composite scoring. If None, uses defaults.
        soft_k: K parameter for soft_numeric_reward exponential decay.
        use_composite: If True, use composite_reward (sum of weighted components).
                      If False, use individual rewards specified below.
        use_exact: Include exact_match_reward (only if use_composite=False)
        use_soft_numeric: Include soft_numeric_reward (only if use_composite=False)
        use_format: Include format_reward (only if use_composite=False)
        use_length: Include length_penalty (only if use_composite=False)

    Returns:
        A TRL-compatible reward function.
    """
    w = weights or RewardWeights()

    if use_composite:
        return _create_composite_reward_func(w, soft_k)

    rewards = []
    if use_exact:
        rewards.append(("exact_match", w.exact_match, exact_match_reward))
    if use_soft_numeric:
        rewards.append(("soft_numeric", w.soft_numeric, soft_numeric_reward))
    if use_format:
        rewards.append(("format", w.format, lambda c, r, **kw: format_reward(c)))
    if use_length:
        rewards.append(("length", w.length, lambda c, **kw: length_penalty(c)))

    return _create_individual_reward_func(rewards, soft_k)


def _create_composite_reward_func(weights: RewardWeights, soft_k: float):
    """Create composite reward function for TRL."""

    def reward_func(
        completions: list[str], reference_answer: list[str], **kwargs: Any
    ) -> list[float]:
        rewards = []
        for completion, reference in zip(completions, reference_answer):
            r = composite_reward(
                completion=_to_text(completion),
                reference=reference,
                weights=weights,
                soft_k=soft_k,
            )
            rewards.append(r)
        return rewards

    return reward_func


def _create_individual_reward_func(
    rewards: list[tuple[str, float, Any]],
    soft_k: float,
):
    """Create individual reward functions that return separate scores."""

    def reward_func(
        completions: list[str], reference_answer: list[str], **kwargs: Any
    ) -> list[float]:
        rewards_list = []
        for completion, reference in zip(completions, reference_answer):
            text = _to_text(completion)
            total = 0.0
            total_weight = 0.0
            for name, weight, fn in rewards:
                if name == "soft_numeric":
                    r = fn(text, reference, k=soft_k)
                elif name == "format":
                    r = fn(text)
                elif name == "length":
                    r = fn(text)
                else:
                    r = fn(text, reference)
                total += weight * r
                total_weight += weight
            rewards_list.append(
                min(total / total_weight, 1.0) if total_weight > 0 else 0.0
            )
        return rewards_list

    return reward_func


def exact_match_reward_func(
    completions: list[str], reference_answer: list[str], **kwargs: Any
) -> list[float]:
    """TRL-compatible exact match reward function.

    Args:
        completions: List of model completions
        reference_answer: List of ground truth answers

    Returns:
        List of rewards (1.0 for exact match, 0.0 otherwise)
    """
    return [exact_match_reward(_to_text(c), r) for c, r in zip(completions, reference_answer)]


def soft_numeric_reward_func(
    completions: list[str],
    reference_answer: list[str],
    k: float = 0.1,
    **kwargs: Any,
) -> list[float]:
    """TRL-compatible soft numeric reward function.

    Args:
        completions: List of model completions
        reference_answer: List of ground truth answers
        k: Exponential decay parameter

    Returns:
        List of rewards (0.0 to 1.0 based on relative error)
    """
    return [
        soft_numeric_reward(_to_text(c), r, k=k) for c, r in zip(completions, reference_answer)
    ]


def format_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    """TRL-compatible format reward function.

    Rewards completions that:
    - Contain "####" trigger (0.4)
    - Have at least 3 lines (0.3)
    - Have numeric content after "####" (0.3)

    Returns:
        List of rewards (0.0 to 1.0)
    """
    return [format_reward(_to_text(c)) for c in completions]


def length_penalty_func(completions: list[str], **kwargs: Any) -> list[float]:
    """TRL-compatible length penalty reward function.

    Penalizes completions that are too short (<20 words) or too long (>512 words).

    Returns:
        List of rewards (0.0 to 1.0)
    """
    return [length_penalty(_to_text(c)) for c in completions]


def create_multi_reward_func(*funcs):
    """Combine multiple reward functions for TRL GRPOTrainer.

    TRL will sum the rewards from each function (or weighted sum if reward_weights is provided).

    Example:
        >>> trainer = GRPOTrainer(
        ...     model="Qwen/Qwen2-0.5B-Instruct",
        ...     reward_funcs=[
        ...         exact_match_reward_func,
        ...         soft_numeric_reward_func,
        ...         format_reward_func,
        ...     ],
        ...     train_dataset=dataset,
        ... )
    """
    return funcs


if __name__ == "__main__":
    test_completions = [
        "To solve this, we first add 2 + 3 = 5.\nThen multiply by 2.\n#### 10",
        "The answer is 10.",
        "Let me calculate...\n#### 10\n\nThat's the answer.",
    ]
    test_references = ["10", "10", "10"]

    print("=== Testing individual reward functions ===\n")

    print("Exact match rewards:")
    rewards = exact_match_reward_func(test_completions, test_references)
    print(f"  {rewards}\n")

    print("Soft numeric rewards (k=0.1):")
    rewards = soft_numeric_reward_func(test_completions, test_references, k=0.1)
    print(f"  {rewards}\n")

    print("Format rewards:")
    rewards = format_reward_func(test_completions)
    print(f"  {rewards}\n")

    print("Length penalty rewards:")
    rewards = length_penalty_func(test_completions)
    print(f"  {rewards}\n")

    print("=== Testing composite reward function ===\n")
    composite = create_reward_func()
    rewards = composite(test_completions, test_references)
    print(f"Composite rewards: {rewards}")

