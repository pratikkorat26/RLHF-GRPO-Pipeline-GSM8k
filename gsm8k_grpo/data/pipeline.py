"""
GSM8K -> GRPO data pipeline with production-style validation and reporting.
"""

import argparse
import json
import logging
import re
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gsm8k_grpo.config.project import PipelineConfig, ProjectConfig
from gsm8k_grpo.config.paths import configure_runtime_environment

warnings.filterwarnings("ignore", category=UserWarning, module="requests")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*chardet.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")

try:
    from datasets import Dataset, DatasetDict, load_dataset

    _DATASETS_AVAILABLE = True
except ModuleNotFoundError:
    Dataset = Any  # type: ignore[assignment]
    DatasetDict = dict  # type: ignore[assignment]
    _DATASETS_AVAILABLE = False

    def load_dataset(*args, **kwargs):
        raise ModuleNotFoundError(
            "The 'datasets' package is required to build or load pipeline artifacts."
        )


try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable, **kwargs):
        return iterable


from gsm8k_grpo.rewards.core import REWARD_CONTRACT_VERSION, reward_contract_dict
from gsm8k_grpo.common.normalization import normalise_numeric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("gsm8k_grpo")

PIPELINE_CONTRACT_VERSION = "v1"
ANSWER_TRIGGER = "####"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful math tutor. "
    "Solve each problem step by step, showing your full reasoning. "
    "At the end, write your final numeric answer after '####'."
)
PUBLIC_SAMPLE_FIELDS = (
    "idx",
    "split",
    "question",
    "reference_answer",
    "prompt_messages",
    "metadata",
)


class PipelineValidationError(RuntimeError):
    """Raised when a pipeline build violates hard validation gates."""


def _require_datasets() -> None:
    if not _DATASETS_AVAILABLE:
        raise ModuleNotFoundError(
            "The 'datasets' package is required for build_pipeline() and dataset serialization."
        )


@dataclass
class GRPOSample:
    idx: int
    split: str
    question: str
    reference_answer: str
    reference_solution: str
    prompt_messages: list[dict]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SplitProcessingResult:
    split: str
    records: list[dict]
    failed_count: int
    failure_reasons: Counter
    raw_count: int


def load_gsm8k(
    split: str = "train",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
) -> Dataset:
    _require_datasets()
    logger.info(
        f"Loading {dataset_name}:{dataset_config} split='{split}' from HuggingFace hub ..."
    )
    ds = load_dataset(dataset_name, dataset_config, split=split)
    logger.info(f"  Loaded {len(ds):,} examples.")
    return ds


def parse_gsm8k_answer(solution: str) -> tuple[str, str]:
    if ANSWER_TRIGGER not in solution:
        raise ValueError(f"No '{ANSWER_TRIGGER}' found in solution: {solution[:80]}")

    cot, _, raw_answer = solution.partition(ANSWER_TRIGGER)
    answer = normalise_numeric(raw_answer.strip())
    if answer is None:
        raise ValueError(f"Cannot parse numeric answer from: '{raw_answer.strip()}'")
    return cot.strip(), answer


def build_prompt_messages(
    question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


def estimate_difficulty(question: str, cot: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", cot)
    n = len([s for s in sentences if s.strip()])
    if n < 5:
        return "easy"
    if n < 10:
        return "medium"
    return "hard"


def _question_fingerprint(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower().strip())


def _prompt_fingerprint(messages: list[dict]) -> str:
    return json.dumps(
        messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _estimated_prompt_length(messages: list[dict]) -> int:
    text = " ".join(m.get("content", "") for m in messages)
    return len(text.split())


def _answer_type(value: str) -> str:
    if "/" in value:
        return "fraction"
    if "." in value:
        return "decimal"
    if value.startswith("-"):
        return "negative_integer"
    return "integer"


def _public_record(record: dict) -> dict:
    return {k: record[k] for k in PUBLIC_SAMPLE_FIELDS}


def validate_record(record: dict, expected_idx: int | None = None) -> list[str]:
    errors: list[str] = []
    missing = [field for field in PUBLIC_SAMPLE_FIELDS if field not in record]
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
        return errors

    if expected_idx is not None and record["idx"] != expected_idx:
        errors.append("unstable_idx")
    if not isinstance(record["question"], str) or not record["question"].strip():
        errors.append("empty_question")
    if (
        not isinstance(record["reference_answer"], str)
        or normalise_numeric(record["reference_answer"]) is None
    ):
        errors.append("invalid_reference_answer")
    if not isinstance(record["metadata"], dict):
        errors.append("invalid_metadata")

    messages = record["prompt_messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        errors.append("invalid_prompt_messages")
        return errors

    expected_roles = ["system", "user"]
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if roles != expected_roles:
        errors.append("invalid_prompt_roles")

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"prompt_message_not_dict:{idx}")
            continue
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            errors.append(f"empty_prompt_content:{idx}")

    return errors


def process_example(
    idx: int,
    split: str,
    raw: dict,
    system_prompt: str,
    add_difficulty: bool,
) -> tuple[dict | None, str | None]:
    try:
        cot, answer = parse_gsm8k_answer(raw["answer"])
    except ValueError as exc:
        return None, f"parse_error:{exc}"

    metadata: dict[str, Any] = {"original_idx": idx}
    if add_difficulty:
        metadata["difficulty"] = estimate_difficulty(raw["question"], cot)

    sample = GRPOSample(
        idx=idx,
        split=split,
        question=raw["question"],
        reference_answer=answer,
        reference_solution=cot,
        prompt_messages=build_prompt_messages(raw["question"], system_prompt),
        metadata=metadata,
    ).to_dict()

    errors = validate_record(sample, expected_idx=idx)
    if errors:
        return None, ";".join(errors)
    return sample, None


def _process_split_serial(
    raw_list: list[dict],
    split: str,
    system_prompt: str,
    add_difficulty: bool,
) -> SplitProcessingResult:
    results: list[dict] = []
    failures: Counter = Counter()
    for idx, raw in enumerate(
        tqdm(raw_list, desc=f"Processing {split} (serial)", unit="ex")
    ):
        out, error = process_example(idx, split, raw, system_prompt, add_difficulty)
        if out is None:
            failures[error or "unknown"] += 1
        else:
            results.append(out)
    return SplitProcessingResult(
        split, results, sum(failures.values()), failures, len(raw_list)
    )


def _process_split_parallel(
    raw_list: list[dict],
    split: str,
    system_prompt: str,
    add_difficulty: bool,
    num_workers: int,
) -> SplitProcessingResult:
    results: list[dict] = []
    failures: Counter = Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                process_example, idx, split, raw, system_prompt, add_difficulty
            ): idx
            for idx, raw in enumerate(raw_list)
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing {split} (parallel x{num_workers})",
            unit="ex",
        ):
            out, error = fut.result()
            if out is None:
                failures[error or "unknown"] += 1
            else:
                results.append(out)
    return SplitProcessingResult(
        split, results, sum(failures.values()), failures, len(raw_list)
    )


def process_split(
    raw_ds: Dataset,
    split: str,
    system_prompt: str,
    add_difficulty: bool,
    num_workers: int,
    parallel_threshold: int = 2_000,
) -> SplitProcessingResult:
    raw_list = list(raw_ds)
    n = len(raw_list)
    use_parallel = (num_workers > 1) and (n >= parallel_threshold)
    logger.info(
        f"  Processing mode: {'parallel x' + str(num_workers) if use_parallel else 'serial'} "
        f"(n={n:,}, threshold={parallel_threshold:,})"
    )

    result = (
        _process_split_parallel(
            raw_list, split, system_prompt, add_difficulty, num_workers
        )
        if use_parallel
        else _process_split_serial(raw_list, split, system_prompt, add_difficulty)
    )
    result.records.sort(key=lambda x: x["idx"])
    logger.info(
        f"  {split}: {len(result.records):,} ok, {result.failed_count} failed "
        f"({result.failed_count / max(n, 1) * 100:.2f}% error rate)"
    )
    return result


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info(f"  Saved {len(records):,} records -> {path}")


def save_hf_dataset(dataset_dict: DatasetDict, path: Path) -> None:
    _require_datasets()
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(path))
    logger.info(f"  Saved HuggingFace DatasetDict -> {path}")


def _write_json_report(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _build_split_report(
    records: list[dict],
    failed_count: int,
    failure_reasons: Counter,
    max_prompt_length: int,
) -> dict:
    difficulty = Counter()
    answer_types = Counter()
    prompt_lengths = []

    for record in records:
        prompt_len = _estimated_prompt_length(record["prompt_messages"])
        prompt_lengths.append(prompt_len)
        answer_types[_answer_type(record["reference_answer"])] += 1
        diff = record.get("metadata", {}).get("difficulty")
        if diff is not None:
            difficulty[diff] += 1

    duplicates_by_prompt = Counter(
        _prompt_fingerprint(r["prompt_messages"]) for r in records
    )
    prompt_duplicate_count = sum(
        count - 1 for count in duplicates_by_prompt.values() if count > 1
    )

    if prompt_lengths:
        length_summary = {
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
            "avg": round(sum(prompt_lengths) / len(prompt_lengths), 2),
            "estimated_truncation_risk_count": sum(
                1 for n in prompt_lengths if n > max_prompt_length
            ),
        }
    else:
        length_summary = {
            "min": 0,
            "max": 0,
            "avg": 0.0,
            "estimated_truncation_risk_count": 0,
        }

    return {
        "record_count": len(records),
        "failed_count": failed_count,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "difficulty_distribution": dict(sorted(difficulty.items())),
        "answer_type_distribution": dict(sorted(answer_types.items())),
        "prompt_length_words": length_summary,
        "duplicate_prompt_count": prompt_duplicate_count,
    }


def _check_cross_split_leakage(hf_splits: dict[str, Dataset]) -> dict:
    split_fps: dict[str, set[str]] = {}
    within_split_duplicates: dict[str, int] = {}
    pairwise_overlaps: dict[str, dict] = {}

    for split, ds in hf_splits.items():
        fingerprints = [_question_fingerprint(row["question"]) for row in ds]
        counts = Counter(fingerprints)
        within_split_duplicates[split] = sum(c - 1 for c in counts.values() if c > 1)
        split_fps[split] = set(fingerprints)

    split_names = list(split_fps.keys())
    found_any = False
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = sorted(split_fps[a] & split_fps[b])
            pairwise_overlaps[f"{a}__{b}"] = {
                "duplicate_count": len(overlap),
                "examples": overlap[:5],
            }
            if overlap:
                found_any = True
                logger.warning(
                    f"  LEAKAGE DETECTED: {len(overlap)} duplicate question(s) between '{a}' and '{b}'"
                )
            else:
                logger.info(f"  No duplicate questions between '{a}' and '{b}'")

    if not found_any:
        logger.info("  Deduplication check passed - all splits are clean.")

    return {
        "within_split_duplicate_questions": within_split_duplicates,
        "pairwise_overlaps": pairwise_overlaps,
    }


def _artifact_paths(out: Path, cfg: PipelineConfig) -> dict[str, Path]:
    return {
        "trainer_jsonl": out / cfg.trainer_artifact_name / "jsonl",
        "trainer_hf": out / cfg.trainer_artifact_name / "hf_dataset",
        "analysis_jsonl": out / cfg.analysis_artifact_name / "jsonl",
        "analysis_hf": out / cfg.analysis_artifact_name / "hf_dataset",
        "reports": out / cfg.reports_dir_name,
    }


def _build_manifest(
    cfg: PipelineConfig,
    split_reports: dict[str, dict],
    paths: dict[str, Path],
) -> dict:
    return {
        "pipeline_contract_version": PIPELINE_CONTRACT_VERSION,
        "reward_contract_version": REWARD_CONTRACT_VERSION,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "source": {
            "dataset_name": cfg.source_dataset_name,
            "dataset_config": cfg.source_dataset_config,
        },
        "config": cfg.to_dict(),
        "public_sample_fields": list(PUBLIC_SAMPLE_FIELDS),
        "artifacts": {name: str(path) for name, path in paths.items()},
        "splits": split_reports,
        "reward_contract": reward_contract_dict(cfg.reward),
    }


def _enforce_quality_gates(
    cfg: PipelineConfig, split_reports: dict[str, dict], leakage_report: dict
) -> None:
    violations: list[str] = []
    for split, report in split_reports.items():
        total = report["record_count"] + report["failed_count"]
        failure_rate = report["failed_count"] / max(total, 1)
        if failure_rate > cfg.max_parse_error_rate:
            violations.append(
                f"{split} parse/drop rate {failure_rate:.4f} exceeds max_parse_error_rate={cfg.max_parse_error_rate:.4f}"
            )
        if report["duplicate_prompt_count"] > 0:
            violations.append(
                f"{split} has {report['duplicate_prompt_count']} duplicate prompt(s)"
            )
        risk_count = report["prompt_length_words"].get(
            "estimated_truncation_risk_count", 0
        )
        risk_rate = risk_count / max(report["record_count"], 1)
        if risk_rate > cfg.max_truncation_risk_rate:
            violations.append(
                f"{split} truncation risk rate {risk_rate:.4f} "
                f"({risk_count}/{report['record_count']} prompts) exceeds "
                f"max_truncation_risk_rate={cfg.max_truncation_risk_rate:.4f}"
            )

    within_split = leakage_report.get("within_split_duplicate_questions", {})
    for split, count in within_split.items():
        if count > 0:
            violations.append(f"{split} has {count} duplicate question(s) within split")

    for pair, details in leakage_report.get("pairwise_overlaps", {}).items():
        if details["duplicate_count"] > 0:
            violations.append(
                f"cross-split leakage detected in {pair}: {details['duplicate_count']} duplicate question(s)"
            )

    if violations:
        raise PipelineValidationError(" | ".join(violations))


def build_pipeline(
    splits: list[str],
    output_dir: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    add_difficulty: bool = True,
    num_workers: int = 4,
    save_jsonl_flag: bool = True,
    save_hf_flag: bool = True,
    max_prompt_length: int = 512,
    max_parse_error_rate: float = 0.01,
    max_truncation_risk_rate: float = 0.0,
    source_dataset_name: str = "openai/gsm8k",
    source_dataset_config: str = "main",
) -> DatasetDict:
    _require_datasets()
    project_defaults = ProjectConfig()
    configure_runtime_environment(storage=project_defaults.storage)
    cfg = replace(
        project_defaults.resolved_pipeline(),
        splits=splits,
        output_dir=output_dir,
        system_prompt=system_prompt,
        add_difficulty=add_difficulty,
        num_workers=num_workers,
        save_jsonl=save_jsonl_flag,
        save_hf=save_hf_flag,
        source_dataset_name=source_dataset_name,
        source_dataset_config=source_dataset_config,
        max_prompt_length=max_prompt_length,
        max_parse_error_rate=max_parse_error_rate,
        max_truncation_risk_rate=max_truncation_risk_rate,
    )

    out = Path(output_dir)
    paths = _artifact_paths(out, cfg)
    trainer_splits: dict[str, Dataset] = {}
    analysis_splits: dict[str, Dataset] = {}
    split_reports: dict[str, dict] = {}

    for split in splits:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Pipeline stage: split='{split}'")
        logger.info(f"{'=' * 60}")

        raw_ds = load_gsm8k(
            split,
            dataset_name=cfg.source_dataset_name,
            dataset_config=cfg.source_dataset_config,
        )
        result = process_split(
            raw_ds,
            split,
            system_prompt,
            add_difficulty,
            num_workers,
            cfg.parallel_threshold,
        )
        public_records = [_public_record(record) for record in result.records]

        split_reports[split] = _build_split_report(
            public_records,
            result.failed_count,
            result.failure_reasons,
            max_prompt_length=cfg.max_prompt_length,
        )

        if save_jsonl_flag:
            save_jsonl(public_records, paths["trainer_jsonl"] / f"{split}.jsonl")
            save_jsonl(result.records, paths["analysis_jsonl"] / f"{split}.jsonl")

        trainer_splits[split] = Dataset.from_list(public_records)
        analysis_splits[split] = Dataset.from_list(result.records)

    trainer_dd = DatasetDict(trainer_splits)
    analysis_dd = DatasetDict(analysis_splits)

    leakage_report = (
        _check_cross_split_leakage(trainer_splits)
        if len(trainer_splits) > 1
        else {
            "within_split_duplicate_questions": {split: 0 for split in trainer_splits},
            "pairwise_overlaps": {},
        }
    )
    _enforce_quality_gates(cfg, split_reports, leakage_report)

    if save_hf_flag:
        save_hf_dataset(trainer_dd, paths["trainer_hf"])
        save_hf_dataset(analysis_dd, paths["analysis_hf"])

    manifest = _build_manifest(cfg, split_reports, paths)
    _write_json_report(manifest, paths["reports"] / "manifest.json")
    _write_json_report(split_reports, paths["reports"] / "validation_summary.json")
    _write_json_report(leakage_report, paths["reports"] / "leakage_report.json")

    logger.info("\nPipeline complete.")
    _print_summary(trainer_dd, split_reports)
    return trainer_dd


def _print_summary(dd: DatasetDict, split_reports: dict[str, dict]) -> None:
    logger.info("\n-- Dataset Summary --------------------------------")
    for split, ds in dd.items():
        logger.info(
            f"  {split:10s}: {len(ds):>6,} examples | columns: {ds.column_names}"
        )
        logger.info(f"    report           : {split_reports.get(split, {})}")
    if "train" in dd and len(dd["train"]) > 0:
        sample = dd["train"][0]
        logger.info("\n-- Example sample (train[0]) ----------------------")
        logger.info(f"  question          : {sample['question'][:80]}...")
        logger.info(f"  reference_answer  : {sample['reference_answer']}")
        logger.info(
            f"  prompt_messages[1]: {str(sample['prompt_messages'][1])[:80]}..."
        )
        if sample["metadata"]:
            logger.info(f"  metadata          : {sample['metadata']}")
    logger.info("---------------------------------------------------\n")


def parse_args() -> argparse.Namespace:
    project_defaults = ProjectConfig()
    defaults = project_defaults.resolved_pipeline()
    p = argparse.ArgumentParser(description="GSM8K -> GRPO data pipeline")
    p.add_argument(
        "--splits",
        nargs="+",
        default=defaults.splits,
        help="Dataset splits to process",
    )
    p.add_argument("--output_dir", default=defaults.output_dir, help="Root output directory")
    p.add_argument(
        "--system_prompt",
        default=defaults.system_prompt,
        help="System prompt for samples",
    )
    p.add_argument(
        "--no_difficulty", action="store_true", help="Skip difficulty estimation"
    )
    p.add_argument(
        "--num_workers", type=int, default=defaults.num_workers, help="Parallel workers for processing"
    )
    p.add_argument("--no_jsonl", action="store_true", help="Skip JSONL output")
    p.add_argument(
        "--no_hf", action="store_true", help="Skip HuggingFace dataset output"
    )
    p.add_argument(
        "--max_prompt_length",
        type=int,
        default=defaults.max_prompt_length,
        help="Estimated prompt length limit for risk stats",
    )
    p.add_argument(
        "--max_parse_error_rate",
        type=float,
        default=defaults.max_parse_error_rate,
        help="Maximum allowed parse/drop rate",
    )
    p.add_argument(
        "--max_truncation_risk_rate",
        type=float,
        default=defaults.max_truncation_risk_rate,
        help="Maximum fraction of prompts allowed to exceed max_prompt_length",
    )
    p.add_argument(
        "--source_dataset_name", default=defaults.source_dataset_name, help="Source dataset name"
    )
    p.add_argument(
        "--source_dataset_config", default=defaults.source_dataset_config, help="Source dataset config"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pipeline(
        splits=args.splits,
        output_dir=args.output_dir,
        system_prompt=args.system_prompt,
        add_difficulty=not args.no_difficulty,
        num_workers=args.num_workers,
        save_jsonl_flag=not args.no_jsonl,
        save_hf_flag=not args.no_hf,
        max_prompt_length=args.max_prompt_length,
        max_parse_error_rate=args.max_parse_error_rate,
        max_truncation_risk_rate=args.max_truncation_risk_rate,
        source_dataset_name=args.source_dataset_name,
        source_dataset_config=args.source_dataset_config,
    )

