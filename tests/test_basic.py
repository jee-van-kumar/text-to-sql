"""Fast, no-GPU-required tests. Run with: pytest -v

These deliberately avoid loading the actual model (slow, needs GPU/network)
so they can run in CI on every push. Model-dependent behavior is covered
separately by evaluate.py, run manually / on a schedule.
"""
import sqlite3

import pytest

from src.config import ProjectConfig
from src.evaluate import build_sqlite_db, normalize_sql, try_execute


def test_default_config_loads():
    cfg = ProjectConfig.load()
    assert cfg.model.base_model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert cfg.lora.r == 16
    assert "q_proj" in cfg.lora.target_modules


def test_config_yaml_override(tmp_path):
    yaml_content = """
train:
  num_train_epochs: 5
lora:
  r: 8
"""
    p = tmp_path / "override.yaml"
    p.write_text(yaml_content)
    cfg = ProjectConfig.load(str(p))
    assert cfg.train.num_train_epochs == 5
    assert cfg.lora.r == 8
    # untouched fields keep their defaults
    assert cfg.model.base_model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_normalize_sql_ignores_case_and_whitespace():
    a = "SELECT  name FROM employees WHERE salary > 100000;"
    b = "select name from employees where salary > 100000"
    assert normalize_sql(a) == normalize_sql(b)


def test_build_sqlite_db_and_execute_simple_schema():
    create_stmt = "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);"
    conn = build_sqlite_db(create_stmt, n_rows=5)
    assert conn is not None

    result, err = try_execute(conn, "SELECT COUNT(*) FROM employees")
    assert err is None
    assert result[0][0] == 5
    conn.close()


def test_try_execute_reports_error_on_bad_sql():
    create_stmt = "CREATE TABLE t (id INT);"
    conn = build_sqlite_db(create_stmt, n_rows=1)
    result, err = try_execute(conn, "SELECT * FROM nonexistent_table")
    assert result is None
    assert err is not None
    conn.close()


def test_build_sqlite_db_handles_malformed_schema_gracefully():
    conn = build_sqlite_db("NOT A VALID CREATE STATEMENT")
    assert conn is None
