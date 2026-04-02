from __future__ import annotations

from pathlib import Path

from project_config import StorageConfig, configure_runtime_environment

PROJECT_STORAGE_ROOT = Path(StorageConfig().project_storage_root)


def storage_root() -> Path:
    return PROJECT_STORAGE_ROOT


def default_pipeline_output_dir() -> str:
    return StorageConfig().pipeline_output_dir


def default_training_dataset_path() -> str:
    return StorageConfig().training_dataset_path


def default_training_output_dir() -> str:
    return StorageConfig().training_output_dir


def default_eval_output_dir() -> str:
    return StorageConfig().eval_output_dir


def default_temp_dir() -> str:
    return StorageConfig().temp_dir


def default_torch_home() -> str:
    return StorageConfig().torch_home


def default_hf_home() -> str:
    return StorageConfig().hf_home


def default_hf_datasets_cache() -> str:
    return StorageConfig().hf_datasets_cache


def default_hf_hub_cache() -> str:
    return StorageConfig().hf_hub_cache


def default_vllm_cache_root() -> str:
    return StorageConfig().vllm_cache_root


def default_triton_cache_dir() -> str:
    return StorageConfig().triton_cache_dir


def default_venv_dir() -> str:
    return StorageConfig().venv_dir


__all__ = [
    "PROJECT_STORAGE_ROOT",
    "configure_runtime_environment",
    "default_eval_output_dir",
    "default_hf_datasets_cache",
    "default_hf_home",
    "default_hf_hub_cache",
    "default_pipeline_output_dir",
    "default_temp_dir",
    "default_torch_home",
    "default_training_dataset_path",
    "default_training_output_dir",
    "default_triton_cache_dir",
    "default_venv_dir",
    "default_vllm_cache_root",
    "storage_root",
]
