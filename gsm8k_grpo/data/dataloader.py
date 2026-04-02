"""
dataloader.py — PyTorch DataLoader for GRPO training
======================================================

Provides:
  - GRPODataset   : wraps the processed HF dataset
  - GRPOCollator  : tokenises prompts and packs into batches
  - build_dataloader : factory with sensible defaults

Design choices for GRPO:
  - Only the *prompt* is tokenised (the model generates completions online)
  - We store reference_answer and metadata for the reward oracle
  - Supports dynamic padding per batch (no static max_length waste)
  - Works with any HuggingFace tokenizer
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("gsm8k_grpo.dataloader")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GRPODataset(Dataset):
    """
    Thin wrapper around a list of GRPOSample dicts.

    Each item exposes:
      question, reference_answer, reference_solution,
      prompt_messages, metadata
    """

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]

    @classmethod
    def from_hf_dataset(cls, hf_ds) -> "GRPODataset":
        """Build from a HuggingFace Dataset object."""
        return cls(list(hf_ds))

    @classmethod
    def from_jsonl(cls, path: str) -> "GRPODataset":
        import json
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records):,} records from {path}")
        return cls(records)

    def difficulty_split(self, difficulty: str) -> "GRPODataset":
        """Filter to a single difficulty bucket (easy/medium/hard)."""
        filtered = [
            r for r in self.records
            if r.get("metadata", {}).get("difficulty") == difficulty
        ]
        return GRPODataset(filtered)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def _messages_to_string(messages: list[dict]) -> str:
    """
    Minimal chat template fallback (used when tokenizer has no apply_chat_template).
    For production, replace with tokenizer.apply_chat_template.
    """
    parts = []
    for m in messages:
        role    = m["role"].upper()
        content = m["content"]
        parts.append(f"[{role}]\n{content}")
    parts.append("[ASSISTANT]")
    return "\n\n".join(parts)


@dataclass
class GRPOCollator:
    """
    Collate a list of GRPOSample dicts into a model-ready batch.

    Returns
    -------
    {
      "input_ids"        : LongTensor [B, L]
      "attention_mask"   : LongTensor [B, L]
      "reference_answers": list[str]   length B
      "questions"        : list[str]   length B
      "metadata"         : list[dict]  length B
    }
    """
    tokenizer: PreTrainedTokenizerBase
    max_prompt_length: int = 512
    padding_side: str = "left"   # left-pad for decoder-only generation

    def __post_init__(self):
        # Some tokenizers don't set a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.warning(
                "tokenizer.pad_token was None → set to eos_token. "
                "Consider setting explicitly."
            )
        self.tokenizer.padding_side = self.padding_side

    def _format_prompt(self, messages: list[dict]) -> str:
        """Apply chat template if available, else fallback."""
        if hasattr(self.tokenizer, "apply_chat_template") and \
                self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return _messages_to_string(messages)

    def __call__(self, samples: list[dict]) -> dict[str, Any]:
        prompts   = [self._format_prompt(s["prompt_messages"]) for s in samples]
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        return {
            "input_ids":         encodings["input_ids"],
            "attention_mask":    encodings["attention_mask"],
            "reference_answers": [s["reference_answer"] for s in samples],
            "questions":         [s["question"] for s in samples],
            "metadata":          [s.get("metadata", {}) for s in samples],
            "prompt_strings":    prompts,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: GRPODataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    max_prompt_length: int = 512,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a ready-to-use DataLoader for GRPO training.

    Parameters
    ----------
    dataset           : GRPODataset
    tokenizer         : HuggingFace tokenizer (must match policy model)
    batch_size        : prompts per step (each generates G completions)
    shuffle           : True for training, False for evaluation
    num_workers       : DataLoader worker processes
    max_prompt_length : truncation length for prompts
    pin_memory        : pin to GPU memory (set False on CPU-only machines)

    Returns
    -------
    torch.utils.data.DataLoader
    """
    collator = GRPOCollator(
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
    logger.info(
        f"DataLoader: {len(dataset):,} samples | "
        f"batch_size={batch_size} | "
        f"steps/epoch={len(loader):,}"
    )
    return loader