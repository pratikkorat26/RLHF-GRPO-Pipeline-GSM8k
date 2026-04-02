from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class StorageConfig:
    project_storage_root: str = os.environ.get(
        "PROJECT_STORAGE_ROOT", "/data/cmpe258-sp24/pratikkorat"
    )

    @property
    def root_path(self) -> Path:
        return Path(self.project_storage_root)

    @property
    def pipeline_output_dir(self) -> str:
        return str(self.root_path / "data" / "grpo")

    @property
    def training_dataset_path(self) -> str:
        return str(self.root_path / "data" / "grpo" / "trainer")

    @property
    def training_output_dir(self) -> str:
        return str(self.root_path / "models" / "grpo")

    @property
    def eval_output_dir(self) -> str:
        return str(self.root_path / "models" / "eval")

    @property
    def temp_dir(self) -> str:
        return str(self.root_path / "tmp")

    @property
    def torch_home(self) -> str:
        return str(self.root_path / ".cache" / "torch")

    @property
    def hf_home(self) -> str:
        return str(self.root_path / ".cache" / "huggingface")

    @property
    def hf_datasets_cache(self) -> str:
        return str(Path(self.hf_home) / "datasets")

    @property
    def hf_hub_cache(self) -> str:
        return str(Path(self.hf_home) / "hub")

    @property
    def vllm_cache_root(self) -> str:
        return str(self.root_path / ".cache" / "vllm")

    @property
    def triton_cache_dir(self) -> str:
        return str(self.root_path / ".cache" / "triton")

    @property
    def venv_dir(self) -> str:
        return str(self.root_path / "venvs" / "gsm8k-grpo")


@dataclass(frozen=True)
class RewardConfig:
    contract_version: str = "v1"
    exact_match_weight: float = 1.0
    soft_numeric_weight: float = 0.3
    format_weight: float = 0.2
    length_weight: float = 0.1
    soft_reward_k: float = 0.1
    min_completion_tokens: int = 20
    max_completion_tokens: int = 512
    answer_extraction_policy: str = "prefer_####_then_last_numeric"
    normalization_policy: str = "currency_commas_percent_fraction_word_numbers"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    splits: list[str] = field(default_factory=lambda: ["train", "test"])
    output_dir: str | None = None
    system_prompt: str = (
        "You are a helpful math tutor. "
        "Solve each problem step by step, showing your full reasoning. "
        "At the end, write your final numeric answer after '####'."
    )
    add_difficulty: bool = True
    num_workers: int = 4
    save_jsonl: bool = True
    save_hf: bool = True
    source_dataset_name: str = "openai/gsm8k"
    source_dataset_config: str = "main"
    max_prompt_length: int = 512
    max_parse_error_rate: float = 0.01
    max_truncation_risk_rate: float = 0.0
    trainer_artifact_name: str = "trainer"
    analysis_artifact_name: str = "analysis"
    reports_dir_name: str = "reports"
    include_reference_solution_in_trainer: bool = False
    parallel_threshold: int = 2_000
    reward: RewardConfig = field(default_factory=RewardConfig)

    def with_storage_defaults(self, storage: StorageConfig) -> "PipelineConfig":
        if self.output_dir is not None:
            return self
        return replace(self, output_dir=storage.pipeline_output_dir)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reward"] = self.reward.to_dict()
        return data


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    dataset_path: str | None = None
    output_dir: str | None = None
    temp_dir: str | None = None
    torch_home: str | None = None
    hf_home: str | None = None
    num_generations: int = 4
    beta: float = 0.04
    use_vllm: bool = True
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.1
    max_prompt_length: int = 512
    max_completion_length: int = 512
    bf16: bool = True
    dataloader_num_workers: int = 0
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int | None = 2
    report_to: str = "none"
    resume_from_checkpoint: str | None = None

    def with_storage_defaults(self, storage: StorageConfig) -> "TrainingConfig":
        return replace(
            self,
            dataset_path=self.dataset_path or storage.training_dataset_path,
            output_dir=self.output_dir or storage.training_output_dir,
            temp_dir=self.temp_dir or storage.temp_dir,
            torch_home=self.torch_home or storage.torch_home,
            hf_home=self.hf_home or storage.hf_home,
        )


@dataclass(frozen=True)
class EvalConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    dataset_path: str | None = None
    split: str = "test"
    output_dir: str | None = None
    temp_dir: str | None = None
    torch_home: str | None = None
    hf_home: str | None = None
    eval_backend: str = "vllm"
    num_samples: int | None = None
    batch_size: int = 32
    max_prompt_length: int = 512
    max_new_tokens: int = 512
    temperature: float = 0.0
    num_workers: int = 4
    seed: int = 42
    gpu_memory_utilization: float = 0.8

    def with_storage_defaults(self, storage: StorageConfig) -> "EvalConfig":
        return replace(
            self,
            dataset_path=self.dataset_path or storage.training_dataset_path,
            output_dir=self.output_dir or storage.eval_output_dir,
            temp_dir=self.temp_dir or storage.temp_dir,
            torch_home=self.torch_home or storage.torch_home,
            hf_home=self.hf_home or storage.hf_home,
        )


@dataclass(frozen=True)
class ProjectConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def resolved_pipeline(self) -> PipelineConfig:
        return replace(
            self.pipeline.with_storage_defaults(self.storage),
            reward=self.reward,
        )

    def resolved_training(self) -> TrainingConfig:
        return self.training.with_storage_defaults(self.storage)

    def resolved_evaluation(self) -> EvalConfig:
        return self.evaluation.with_storage_defaults(self.storage)


def configure_runtime_environment(
    *,
    temp_dir: str | None = None,
    torch_home: str | None = None,
    hf_home: str | None = None,
    storage: StorageConfig | None = None,
) -> dict[str, str]:
    resolved_storage = storage or StorageConfig()
    resolved_temp_dir = Path(temp_dir or resolved_storage.temp_dir)
    resolved_torch_home = Path(torch_home or resolved_storage.torch_home)
    resolved_hf_home = Path(hf_home or resolved_storage.hf_home)
    resolved_hf_datasets_cache = resolved_hf_home / "datasets"
    resolved_hf_hub_cache = resolved_hf_home / "hub"
    resolved_vllm_cache_root = Path(resolved_storage.vllm_cache_root)
    resolved_triton_cache_dir = Path(resolved_storage.triton_cache_dir)
    resolved_xdg_cache_home = resolved_storage.root_path / ".cache"

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
        "PROJECT_STORAGE_ROOT": resolved_storage.project_storage_root,
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
