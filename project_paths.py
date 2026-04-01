from __future__ import annotations

import os
from pathlib import Path

PROJECT_STORAGE_ROOT = Path(
    os.environ.get("PROJECT_STORAGE_ROOT", "/data/cmpe258-sp24/pratikkorat")
)


def storage_root() -> Path:
    return PROJECT_STORAGE_ROOT


def default_pipeline_output_dir() -> str:
    return str(storage_root() / "data" / "grpo")


def default_training_dataset_path() -> str:
    return str(storage_root() / "data" / "grpo" / "trainer")


def default_training_output_dir() -> str:
    return str(storage_root() / "models" / "grpo")


def default_eval_output_dir() -> str:
    return str(storage_root() / "models" / "eval")


def default_temp_dir() -> str:
    return str(storage_root() / "tmp")


def default_torch_home() -> str:
    return str(storage_root() / ".cache" / "torch")


def default_hf_home() -> str:
    return str(storage_root() / ".cache" / "huggingface")


def default_hf_datasets_cache() -> str:
    return str(Path(default_hf_home()) / "datasets")


def default_hf_hub_cache() -> str:
    return str(Path(default_hf_home()) / "hub")


def default_vllm_cache_root() -> str:
    return str(storage_root() / ".cache" / "vllm")


def default_triton_cache_dir() -> str:
    return str(storage_root() / ".cache" / "triton")


def default_venv_dir() -> str:
    return str(storage_root() / "venvs" / "gsm8k-grpo")


def configure_runtime_environment(
    *,
    temp_dir: str | None = None,
    torch_home: str | None = None,
    hf_home: str | None = None,
) -> dict[str, str]:
    resolved_temp_dir = Path(temp_dir or default_temp_dir())
    resolved_torch_home = Path(torch_home or default_torch_home())
    resolved_hf_home = Path(hf_home or default_hf_home())
    resolved_hf_datasets_cache = resolved_hf_home / "datasets"
    resolved_hf_hub_cache = resolved_hf_home / "hub"
    resolved_vllm_cache_root = Path(default_vllm_cache_root())
    resolved_triton_cache_dir = Path(default_triton_cache_dir())
    resolved_xdg_cache_home = storage_root() / ".cache"

    for path in (
        resolved_temp_dir,
        resolved_xdg_cache_home,
        resolved_torch_home,
        resolved_hf_home,
        resolved_hf_datasets_cache,
        resolved_hf_hub_cache,
        resolved_vllm_cache_root,
        resolved_triton_cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    env_updates = {
        "TMPDIR": str(resolved_temp_dir),
        "TMP": str(resolved_temp_dir),
        "TEMP": str(resolved_temp_dir),
        "XDG_CACHE_HOME": str(resolved_xdg_cache_home),
        "TORCH_HOME": str(resolved_torch_home),
        "HF_HOME": str(resolved_hf_home),
        "HUGGINGFACE_HUB_CACHE": str(resolved_hf_hub_cache),
        "HF_DATASETS_CACHE": str(resolved_hf_datasets_cache),
        "VLLM_CACHE_ROOT": str(resolved_vllm_cache_root),
        "TRITON_CACHE_DIR": str(resolved_triton_cache_dir),
    }
    os.environ.update(env_updates)
    return env_updates
