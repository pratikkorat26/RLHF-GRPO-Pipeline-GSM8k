import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from datasets import Dataset, load_from_disk
    DATASETS_AVAILABLE = True
except ModuleNotFoundError:
    Dataset = None
    load_from_disk = None
    DATASETS_AVAILABLE = False

from config import RewardConfig
from pipeline import (
    _check_cross_split_leakage,
    build_pipeline,
    build_prompt_messages,
    estimate_difficulty,
    parse_gsm8k_answer,
    validate_record,
    PipelineValidationError,
    PUBLIC_SAMPLE_FIELDS,
)
from utils import normalise_numeric
from reward import (
    ANSWER_EXTRACTION_POLICY,
    NORMALIZATION_POLICY,
    REWARD_CONTRACT_VERSION,
    RewardWeights,
    composite_reward,
    compute_group_rewards,
    compute_grpo_advantages,
    exact_match_reward,
    extract_final_answer,
    format_reward,
    reward_contract_dict,
    soft_numeric_reward,
)


class TestNormaliseNumeric(unittest.TestCase):
    def test_plain_integer(self):
        self.assertEqual(normalise_numeric("42"), "42")

    def test_negative(self):
        self.assertEqual(normalise_numeric("-15"), "-15")

    def test_thousands_comma(self):
        self.assertEqual(normalise_numeric("1,024"), "1024")

    def test_currency_dollar(self):
        self.assertEqual(normalise_numeric("$1,200"), "1200")

    def test_percentage(self):
        self.assertEqual(normalise_numeric("35%"), "35")

    def test_decimal(self):
        self.assertEqual(normalise_numeric("3.14"), "3.14")

    def test_fraction_half(self):
        self.assertEqual(normalise_numeric("1/2"), "0.5")

    def test_word_number(self):
        self.assertEqual(normalise_numeric("twelve"), "12")

    def test_none_on_garbage(self):
        self.assertIsNone(normalise_numeric("no number here"))


class TestParsing(unittest.TestCase):
    def test_basic_parse(self):
        cot, ans = parse_gsm8k_answer("Work it out.\n#### 8")
        self.assertEqual(cot, "Work it out.")
        self.assertEqual(ans, "8")

    def test_missing_trigger_raises(self):
        with self.assertRaises(ValueError):
            parse_gsm8k_answer("No marker")

    def test_prompt_messages_structure(self):
        msgs = build_prompt_messages("What is 2+2?")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])

    def test_difficulty(self):
        self.assertEqual(estimate_difficulty("q", "Step one. Step two."), "easy")


class TestRecordValidation(unittest.TestCase):
    def test_validate_record_ok(self):
        record = {
            "idx": 0,
            "split": "train",
            "question": "What is 2+2?",
            "reference_answer": "4",
            "prompt_messages": build_prompt_messages("What is 2+2?"),
            "metadata": {"original_idx": 0},
        }
        self.assertEqual(validate_record(record, expected_idx=0), [])

    def test_validate_record_missing_field(self):
        record = {
            "idx": 0,
            "split": "train",
            "question": "What is 2+2?",
            "reference_answer": "4",
            "metadata": {},
        }
        errors = validate_record(record)
        self.assertTrue(any(err.startswith("missing_fields:") for err in errors))

    def test_validate_record_prompt_roles(self):
        record = {
            "idx": 0,
            "split": "train",
            "question": "What is 2+2?",
            "reference_answer": "4",
            "prompt_messages": [{"role": "user", "content": "bad"}],
            "metadata": {},
        }
        self.assertIn("invalid_prompt_messages", validate_record(record))


class TestRewardFunctions(unittest.TestCase):
    def test_exact_match_correct(self):
        self.assertEqual(exact_match_reward("blah #### 42", "42"), 1.0)

    def test_exact_match_wrong(self):
        self.assertEqual(exact_match_reward("blah #### 43", "42"), 0.0)

    def test_soft_reward_perfect(self):
        self.assertAlmostEqual(soft_numeric_reward("#### 100", "100"), 1.0, places=5)

    def test_format_reward_full(self):
        comp = "Step 1.\nStep 2.\nStep 3.\n#### 42"
        self.assertAlmostEqual(format_reward(comp), 1.0, places=5)

    def test_extract_fallback_last_numeric(self):
        self.assertEqual(extract_final_answer("We tried 40, then 41, final answer 42"), "42")

    def test_extract_prefers_marker(self):
        self.assertEqual(extract_final_answer("Maybe 99.\n#### 42\nActually 100"), "42")

    def test_extract_fraction_after_marker(self):
        self.assertEqual(extract_final_answer("The fraction is #### 3/4"), "0.75")

    def test_extract_none_on_no_number(self):
        self.assertIsNone(extract_final_answer("I refuse to answer."))

    def test_composite_reward_penalises_format_only_answer(self):
        format_only = "Step 1.\nStep 2.\nStep 3.\n#### 999"
        correct = "Step 1.\nStep 2.\nStep 3.\n#### 42"
        self.assertLess(composite_reward(format_only, "42"), composite_reward(correct, "42"))

    def test_composite_reward_penalises_last_number_trap(self):
        bad = "I considered 42 first, but final answer is 100"
        good = "I considered 100 first, but final answer is 42"
        self.assertLess(composite_reward(bad, "42"), composite_reward(good, "42"))

    def test_reward_contract_metadata(self):
        payload = reward_contract_dict(RewardConfig())
        self.assertEqual(payload["contract_version"], REWARD_CONTRACT_VERSION)
        self.assertEqual(payload["answer_extraction_policy"], ANSWER_EXTRACTION_POLICY)
        self.assertEqual(payload["normalization_policy"], NORMALIZATION_POLICY)

    def test_reward_weights_from_config(self):
        weights = RewardWeights.from_config(RewardConfig(format_weight=0.5))
        self.assertEqual(weights.format, 0.5)


class TestGRPOAdvantages(unittest.TestCase):
    def test_advantage_zero_mean(self):
        advs = compute_grpo_advantages([0.2, 0.5, 0.8, 1.0, 0.0])
        self.assertAlmostEqual(sum(advs) / len(advs), 0.0, places=5)

    def test_constant_rewards_zero_advantages(self):
        advs = compute_grpo_advantages([0.5] * 5)
        self.assertTrue(all(abs(a) < 1e-2 for a in advs))

    def test_group_rewards_length(self):
        rewards = compute_group_rewards(["#### 1", "#### 2", "#### 3"], "1")
        self.assertEqual(len(rewards), 3)
        self.assertGreater(rewards[0], rewards[1])


@unittest.skipUnless(DATASETS_AVAILABLE, "datasets package is required for dataset-backed tests")
class TestLeakageChecks(unittest.TestCase):
    def _make_fake_ds(self, questions: list[str]):
        return Dataset.from_list([{"question": q} for q in questions])

    def test_clean_splits_no_overlap(self):
        report = _check_cross_split_leakage(
            {
                "train": self._make_fake_ds(["What is 2+2?", "Solve for x."]),
                "test": self._make_fake_ds(["What is 3+3?", "Find area."]),
            }
        )
        self.assertEqual(report["pairwise_overlaps"]["train__test"]["duplicate_count"], 0)

    def test_duplicate_overlap_reported(self):
        report = _check_cross_split_leakage(
            {
                "train": self._make_fake_ds(["What is 2+2?", "Unique."]),
                "test": self._make_fake_ds(["What is 2+2?", "Different."]),
            }
        )
        self.assertEqual(report["pairwise_overlaps"]["train__test"]["duplicate_count"], 1)


@unittest.skipUnless(DATASETS_AVAILABLE, "datasets package is required for build tests")
class TestBuildPipeline(unittest.TestCase):
    def setUp(self):
        self.train_rows = [
            {
                "question": "What is 2 + 2?",
                "answer": "Add them.\n#### 4",
            },
            {
                "question": "What is 10 / 2?",
                "answer": "Divide the number.\n#### 5",
            },
        ]
        self.test_rows = [
            {
                "question": "What is 3 + 3?",
                "answer": "Add them.\n#### 6",
            }
        ]

    def _fake_loader(self, split, dataset_name="openai/gsm8k", dataset_config="main"):
        rows = self.train_rows if split == "train" else self.test_rows
        return Dataset.from_list(rows)

    def test_build_pipeline_writes_artifacts_and_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pipeline.load_gsm8k", side_effect=self._fake_loader):
                dd = build_pipeline(
                    splits=["train", "test"],
                    output_dir=tmpdir,
                    num_workers=1,
                    save_jsonl_flag=True,
                    save_hf_flag=True,
                )

            self.assertEqual(set(dd.keys()), {"train", "test"})
            trainer_jsonl = Path(tmpdir) / "trainer" / "jsonl" / "train.jsonl"
            analysis_jsonl = Path(tmpdir) / "analysis" / "jsonl" / "train.jsonl"
            manifest_path = Path(tmpdir) / "reports" / "manifest.json"
            validation_path = Path(tmpdir) / "reports" / "validation_summary.json"
            leakage_path = Path(tmpdir) / "reports" / "leakage_report.json"

            self.assertTrue(trainer_jsonl.exists())
            self.assertTrue(analysis_jsonl.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(validation_path.exists())
            self.assertTrue(leakage_path.exists())

            trainer_lines = [json.loads(line) for line in trainer_jsonl.read_text(encoding="utf-8").splitlines()]
            analysis_lines = [json.loads(line) for line in analysis_jsonl.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(set(trainer_lines[0].keys()), set(PUBLIC_SAMPLE_FIELDS))
            self.assertIn("reference_solution", analysis_lines[0])
            self.assertNotIn("reference_solution", trainer_lines[0])

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline_contract_version"], "v1")
            self.assertEqual(manifest["reward_contract"]["contract_version"], REWARD_CONTRACT_VERSION)

            trainer_hf = load_from_disk(str(Path(tmpdir) / "trainer" / "hf_dataset"))
            analysis_hf = load_from_disk(str(Path(tmpdir) / "analysis" / "hf_dataset"))
            self.assertNotIn("reference_solution", trainer_hf["train"].column_names)
            self.assertIn("reference_solution", analysis_hf["train"].column_names)

    def test_build_pipeline_rejects_cross_split_leakage(self):
        def dup_loader(split, dataset_name="openai/gsm8k", dataset_config="main"):
            rows = [{"question": "Same question?", "answer": "Work.\n#### 1"}]
            return Dataset.from_list(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pipeline.load_gsm8k", side_effect=dup_loader):
                with self.assertRaises(PipelineValidationError):
                    build_pipeline(
                        splits=["train", "test"],
                        output_dir=tmpdir,
                        num_workers=1,
                    )

    def test_build_pipeline_rejects_parse_error_threshold(self):
        def bad_loader(split, dataset_name="openai/gsm8k", dataset_config="main"):
            rows = [{"question": "Broken", "answer": "No marker"}]
            return Dataset.from_list(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pipeline.load_gsm8k", side_effect=bad_loader):
                with self.assertRaises(PipelineValidationError):
                    build_pipeline(
                        splits=["train"],
                        output_dir=tmpdir,
                        num_workers=1,
                        max_parse_error_rate=0.0,
                    )

    def test_build_pipeline_rejects_truncation_risk(self):
        long_rows = [{"question": "word " * 600, "answer": "Math.\n#### 1"}]

        def long_loader(split, **_):
            return Dataset.from_list(long_rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pipeline.load_gsm8k", side_effect=long_loader):
                with self.assertRaises(PipelineValidationError):
                    build_pipeline(
                        splits=["train"],
                        output_dir=tmpdir,
                        num_workers=1,
                        max_prompt_length=100,
                        max_truncation_risk_rate=0.0,
                    )

    def test_build_pipeline_is_logically_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "run1"
            second = Path(tmpdir) / "run2"
            with patch("pipeline.load_gsm8k", side_effect=self._fake_loader):
                build_pipeline(["train", "test"], str(first), num_workers=1)
            with patch("pipeline.load_gsm8k", side_effect=self._fake_loader):
                build_pipeline(["train", "test"], str(second), num_workers=1)

            first_train = (first / "trainer" / "jsonl" / "train.jsonl").read_text(encoding="utf-8")
            second_train = (second / "trainer" / "jsonl" / "train.jsonl").read_text(encoding="utf-8")
            self.assertEqual(first_train, second_train)

            first_validation = json.loads((first / "reports" / "validation_summary.json").read_text(encoding="utf-8"))
            second_validation = json.loads((second / "reports" / "validation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(first_validation, second_validation)


if __name__ == "__main__":
    unittest.main()
