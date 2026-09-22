"""Dataset loading, prompt construction, and formatting for SFT."""
from __future__ import annotations

from typing import Dict, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    schema: str,
    question: str,
    system_prompt: str,
    add_generation_prompt: bool = True,
) -> str:
    """Build a chat-template prompt for inference (no assistant answer)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def _format_row(row: Dict, tokenizer: PreTrainedTokenizerBase, system_prompt: str) -> Dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Schema:\n{row['context']}\n\nQuestion: {row['question']}",
        },
        {"role": "assistant", "content": row["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def load_and_prepare(
    tokenizer: PreTrainedTokenizerBase, cfg: DataConfig
) -> Tuple[Dataset, Dataset]:
    """Load b-mc2/sql-create-context, subsample, split, and format for SFT.

    Returns (train_dataset, eval_dataset) each with a single "text" column.
    Raw context/question/answer columns are preserved under a parallel
    "raw" split when you need them for evaluation (see evaluate.py).
    """
    raw = load_dataset(cfg.dataset_id, split="train")
    raw = raw.shuffle(seed=cfg.seed).select(range(min(cfg.train_size, len(raw))))
    split = raw.train_test_split(test_size=cfg.test_split, seed=cfg.seed)
    train_raw, eval_raw = split["train"], split["test"]

    train_ds = train_raw.map(
        lambda r: _format_row(r, tokenizer, cfg.system_prompt),
        remove_columns=[],  # keep raw columns too; SFTTrainer only needs "text"
    )
    eval_ds = eval_raw.map(
        lambda r: _format_row(r, tokenizer, cfg.system_prompt),
        remove_columns=[],
    )
    return train_ds, eval_ds


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dataset:
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)

    cols_to_drop = [c for c in dataset.column_names if c != "text"]
    return dataset.map(_tok, batched=True, remove_columns=dataset.column_names)
