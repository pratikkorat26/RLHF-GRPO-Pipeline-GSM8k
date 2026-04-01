"""
Model and tokenizer loading utilities.

Single responsibility: load a causal LM and its tokenizer with the
settings required for GRPO (left-padding, pad-token fallback, dtype).
All other concerns (training loop, evaluation) live elsewhere.
"""

from __future__ import annotations

import logging
import sys
import types

import torch

# ---------------------------------------------------------------------------
# Torchvision ABI guard
# ---------------------------------------------------------------------------
# Some virtualenvs (e.g. vllm-engine) ship a torchvision build that is ABI-
# incompatible with the installed torch.  Loading it raises:
#   RuntimeError: operator torchvision::nms does not exist
# Transformers' image_utils.py calls torchvision.transforms at import time,
# which triggers the crash even for pure text models.
# Fix: if torchvision is broken, replace it with an empty stub so that
# `transformers.utils.is_torchvision_available()` returns False and the
# vision import path is never taken.
if "torchvision" not in sys.modules:
    try:
        import torchvision as _tv  # noqa: F401
    except (RuntimeError, OSError):
        # Insert a minimal stub so subsequent `import torchvision` succeeds
        # without triggering the broken native extension.
        _stub = types.ModuleType("torchvision")
        _stub.__version__ = "0.0.0+stub"
        sys.modules["torchvision"] = _stub
        sys.modules["torchvision.transforms"] = types.ModuleType("torchvision.transforms")

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("gsm8k_grpo.model")


def _best_attn_impl() -> str:
    """Return flash_attention_2 if flash-attn is installed, else eager."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a tokenizer with padding defaults suitable for text generation."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token")
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_and_tokenizer(
    model_name: str,
    device_map: str | dict | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
    training: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and tokenizer ready for GRPO training or evaluation.

    Args:
        model_name: HuggingFace model ID or local path.
        device_map: Explicit transformers device map. None keeps placement local.
        torch_dtype: Weight precision. bfloat16 recommended for modern GPUs.
        attn_implementation: "flash_attention_2" or "eager". None = auto-detect
            (uses flash_attention_2 if flash-attn is installed, else eager).
        training: When True, prefer the single visible CUDA device.

    Returns:
        (model, tokenizer) tuple.
    """
    impl = attn_implementation or _best_attn_impl()
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    effective_dtype = torch_dtype if torch.cuda.is_available() else torch.float32

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Qwen base models ship without a dedicated pad token — use eos instead.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token")

    # Left-padding is required for decoder-only generation so that all
    # completions in a batch start at the same relative position.
    tokenizer.padding_side = "left"

    load_kwargs = {
        "torch_dtype": effective_dtype,
        "attn_implementation": impl,
    }
    effective_device_map = device_map
    if effective_device_map is None and training and torch.cuda.is_available():
        effective_device_map = {"": 0}
    if effective_device_map is not None:
        load_kwargs["device_map"] = effective_device_map

    logger.info(
        "Loading model: %s  dtype=%s  attn=%s  device_map=%s  target_device=%s",
        model_name,
        effective_dtype,
        impl,
        effective_device_map,
        target_device,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if effective_device_map is None:
        model = model.to(target_device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded — {n_params:.1f}M parameters")
    return model, tokenizer
