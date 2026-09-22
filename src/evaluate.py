"""Evaluate the fine-tuned model with two metrics:

1. Exact match  - normalized string comparison of predicted vs gold SQL.
2. Execution accuracy - build a throwaway SQLite DB from the CREATE TABLE
   schema, populate it with a few synthetic rows, run both the gold and
   predicted SQL against it, and compare result sets. This is the metric
   that actually matters for text-to-SQL: two queries can differ in text
   but return identical results (or vice versa).

Usage:
    python -m src.evaluate --config configs/default.yaml \
        --adapter-dir ./artifacts/adapter --n 200
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ProjectConfig
from .data import build_prompt, load_and_prepare

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    n_examples: int
    exact_match_accuracy: float
    execution_accuracy: float
    execution_error_rate: float  # fraction where predicted SQL failed to execute at all


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def _extract_column_type(col_def: str) -> str:
    """Very small heuristic to pick a SQLite-friendly placeholder value type."""
    col_def_lower = col_def.lower()
    if any(t in col_def_lower for t in ["int", "float", "double", "real", "decimal", "numeric"]):
        return "number"
    return "text"


def build_sqlite_db(create_stmt: str, n_rows: int = 5) -> Optional[sqlite3.Connection]:
    """Create an in-memory SQLite DB from a CREATE TABLE statement and
    populate it with synthetic rows so execution comparisons are possible.
    Returns None if the schema can't be parsed/executed (skipped in metrics).
    """
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    try:
        cur.execute(create_stmt)
    except sqlite3.Error as e:
        logger.debug("Could not create table: %s", e)
        conn.close()
        return None

    match = re.search(r"CREATE TABLE\s+([`\"\[]?\w+[`\"\]]?)\s*\((.+)\)", create_stmt, re.S | re.I)
    if not match:
        return conn  # table created but couldn't parse columns; DB still usable, just empty
    table_name = match.group(1).strip("`\"[]")
    cols_raw = match.group(2)

    col_names, col_types = [], []
    depth = 0
    current = ""
    parts = []
    for ch in cols_raw:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)

    for part in parts:
        part = part.strip()
        if not part or part.upper().startswith(("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT")):
            continue
        tokens = part.split()
        if not tokens:
            continue
        col_names.append(tokens[0].strip("`\"[]"))
        col_types.append(_extract_column_type(part))

    if col_names:
        placeholders = ",".join(["?"] * len(col_names))
        insert_sql = f"INSERT INTO {table_name} ({','.join(col_names)}) VALUES ({placeholders})"
        for i in range(n_rows):
            row = [
                (i + 1) if t == "number" else f"val_{i+1}"
                for t in col_types
            ]
            try:
                cur.execute(insert_sql, row)
            except sqlite3.Error as e:
                logger.debug("Row insert failed (schema too complex to synth-fill): %s", e)
                break
    conn.commit()
    return conn


def try_execute(conn: sqlite3.Connection, sql: str):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return sorted(cur.fetchall()), None
    except sqlite3.Error as e:
        return None, str(e)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model_for_eval(cfg: ProjectConfig, adapter_dir: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or cfg.model.base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def run_evaluation(
    cfg: ProjectConfig, adapter_dir: Optional[str], n: int, max_new_tokens: int
) -> EvalResult:
    model, tokenizer = load_model_for_eval(cfg, adapter_dir)
    _, eval_ds = load_and_prepare(tokenizer, cfg.data)
    n = min(n, len(eval_ds))

    exact_matches = 0
    exec_matches = 0
    exec_errors = 0
    per_example: List[dict] = []

    for i in range(n):
        row = eval_ds[i]
        prompt = build_prompt(tokenizer, row["context"], row["question"], cfg.data.system_prompt)
        pred = generate(model, tokenizer, prompt, max_new_tokens)
        gold = row["answer"]

        em = normalize_sql(pred) == normalize_sql(gold)
        exact_matches += int(em)

        conn = build_sqlite_db(row["context"])
        ex_match = False
        pred_err = None
        if conn is not None:
            gold_result, gold_err = try_execute(conn, gold)
            pred_result, pred_err = try_execute(conn, pred)
            if pred_err is not None:
                exec_errors += 1
            elif gold_err is None and gold_result == pred_result:
                ex_match = True
            conn.close()
        exec_matches += int(ex_match)

        per_example.append(
            {
                "question": row["question"],
                "gold": gold,
                "pred": pred,
                "exact_match": em,
                "execution_match": ex_match,
                "exec_error": pred_err,
            }
        )

    result = EvalResult(
        n_examples=n,
        exact_match_accuracy=exact_matches / n,
        execution_accuracy=exec_matches / n,
        execution_error_rate=exec_errors / n,
    )
    return result, per_example


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the SQL-LoRA model")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--adapter-dir", type=str, default=None, help="Path to LoRA adapter; omit to eval base model")
    p.add_argument("--n", type=int, default=100, help="Number of eval examples")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output", type=str, default="eval_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig.load(args.config)
    result, per_example = run_evaluation(cfg, args.adapter_dir, args.n, args.max_new_tokens)

    logger.info("Exact match accuracy:    %.2f%%", result.exact_match_accuracy * 100)
    logger.info("Execution accuracy:      %.2f%%", result.execution_accuracy * 100)
    logger.info("Execution error rate:    %.2f%%", result.execution_error_rate * 100)

    with open(args.output, "w") as f:
        json.dump({"summary": asdict(result), "examples": per_example}, f, indent=2)
    logger.info("Wrote detailed results to %s", args.output)


if __name__ == "__main__":
    main()
