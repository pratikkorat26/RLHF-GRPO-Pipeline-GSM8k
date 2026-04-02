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
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Torchvision ABI guard
# ---------------------------------------------------------------------------
# Some virtualenvs (e.g. vllm-engine) ship a torchvision build that is ABI-
# incompatible with the installed torch. Loading it raises:
#   RuntimeError: operator torchvision::nms does not exist
# Transformers' image_utils.py calls torchvision.transforms at import time,
# which triggers the crash even for pure text models.
if "torchvision" not in sys.modules:
    try:
        import torchvision as _tv  # noqa: F401
    except (RuntimeError, OSError):
        _stub = types.ModuleType("torchvision")
        _stub.__version__ = "0.0.0+stub"
        sys.modules["torchvision"] = _stub
        sys.modules["torchvision.transforms"] = types.ModuleType("torchvision.transforms")

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("gsm8k_grpo.model")

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _best_attn_impl() -> str:
    """Return flash_attention_2 if flash-attn is installed, else eager."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


def _looks_like_local_path(model_name: str) -> bool:
    path = Path(model_name)
    if path.is_absolute() or path.exists():
        return True
    return model_name.startswith(("./", "../", ".\\", "..\\"))


def _resolve_model_source(model_name: str) -> tuple[str, Path | None]:
    if not _looks_like_local_path(model_name):
        return model_name, None

    resolved = Path(model_name).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {resolved}. "
            "Pass a valid local checkpoint/model directory or a Hugging Face model id."
        )
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Local model path is not a directory: {resolved}. "
            "Pass a saved model/checkpoint directory."
        )
    return str(resolved), resolved


def _require_local_files(model_dir: Path, required: tuple[str, ...], kind: str) -> None:
    if any((model_dir / name).exists() for name in required):
        return
    raise FileNotFoundError(
        f"Local {kind} directory is missing expected files under {model_dir}. "
        f"Expected one of: {', '.join(required)}"
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a tokenizer with padding defaults suitable for text generation."""
    source, local_dir = _resolve_model_source(model_name)
    if local_dir is not None:
        _require_local_files(local_dir, _TOKENIZER_FILES, "tokenizer")

    logger.info("Loading tokenizer: %s", source)
    tokenizer = AutoTokenizer.from_pretrained(source)
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
    """Load a causal LM and tokenizer ready for GRPO training or evaluation."""
    source, local_dir = _resolve_model_source(model_name)
    if local_dir is not None:
        _require_local_files(local_dir, _TOKENIZER_FILES, "tokenizer")
        _require_local_files(local_dir, _MODEL_FILES, "model")

    impl = attn_implementation or _best_attn_impl()
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    effective_dtype = torch_dtype if torch.cuda.is_available() else torch.float32
    tokenizer = load_tokenizer(source)

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
        source,
        effective_dtype,
        impl,
        effective_device_map,
        target_device,
    )
    model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
    if effective_device_map is None:
        model = model.to(target_device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded — %.1fM parameters", n_params)
    return model, tokenizer
