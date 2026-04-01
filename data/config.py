"""
Central configuration for the GSM8K GRPO data and reward pipeline.
"""
from dataclasses import asdict, dataclass, field

from project_paths import default_pipeline_output_dir


@dataclass(frozen=True)
class RewardConfig:
    contract_version: str = "v1"
    exact_match_weight: float = 1.0
    soft_numeric_weight: float = 0.3
    format_weight: float = 0.2
    length_weight: float = 0.1
    soft_reward_k: float = 0.1
    min_completion_tokens: int = 20   # length_penalty lower bound
    max_completion_tokens: int = 512  # length_penalty upper bound
    answer_extraction_policy: str = "prefer_####_then_last_numeric"
    normalization_policy: str = "currency_commas_percent_fraction_word_numbers"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    splits: list[str] = field(default_factory=lambda: ["train", "test"])
    output_dir: str = field(default_factory=default_pipeline_output_dir)
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
    max_truncation_risk_rate: float = 0.0  # max fraction of prompts allowed to exceed max_prompt_length
    trainer_artifact_name: str = "trainer"
    analysis_artifact_name: str = "analysis"
    reports_dir_name: str = "reports"
    include_reference_solution_in_trainer: bool = False
    parallel_threshold: int = 2_000  # minimum dataset size to use ProcessPoolExecutor
    reward: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reward"] = self.reward.to_dict()
        return data
