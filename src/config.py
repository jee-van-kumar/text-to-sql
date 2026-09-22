"""Central configuration for the text-to-SQL LoRA project.

All scripts (train.py, evaluate.py, inference.py, api/main.py) read from
here or from a YAML file that overrides these defaults. Keeping config in
one typed place means no more hardcoded constants scattered across cells.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    base_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_dir: str = "./artifacts/adapter"
    merged_dir: str = "./artifacts/merged"
    max_seq_length: int = 512


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class DataConfig:
    dataset_id: str = "b-mc2/sql-create-context"
    train_size: int = 3000
    test_split: float = 0.05
    seed: int = 42
    system_prompt: str = (
        "You are a SQL assistant. Given a table schema and a question, "
        "reply with ONLY the SQL query, nothing else."
    )


@dataclass
class TrainConfig:
    output_dir: str = "./tinyllama-sql-lora"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 10
    logging_steps: int = 10
    eval_steps: int = 50
    fp16: bool = True
    bf16: bool = False
    report_to: str = "none"  # set to "wandb" to enable experiment tracking
    run_name: Optional[str] = None
    seed: int = 42


@dataclass
class ServeConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_new_tokens: int = 128
    use_merged_weights: bool = True  # False -> load base + adapter separately


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ProjectConfig":
        cfg = cls()
        if path and os.path.exists(path):
            with open(path) as f:
                overrides = yaml.safe_load(f) or {}
            for section, values in overrides.items():
                if hasattr(cfg, section) and isinstance(values, dict):
                    sub = getattr(cfg, section)
                    for k, v in values.items():
                        if hasattr(sub, k):
                            setattr(sub, k, v)
        return cfg


DEFAULT_CONFIG_PATH = "configs/default.yaml"
