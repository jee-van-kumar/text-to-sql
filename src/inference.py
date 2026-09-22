"""Inference utilities: merge LoRA weights and provide a simple predictor
class used by both the CLI and the FastAPI service.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ProjectConfig
from .data import build_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_and_save(cfg: ProjectConfig) -> str:
    """Merge the LoRA adapter into the base model weights and save the
    result. Merged weights load faster at serve time (no PEFT overhead)
    at the cost of extra disk space.
    """
    logger.info("Loading base model %s", cfg.model.base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    logger.info("Loading adapter from %s", cfg.model.adapter_dir)
    model = PeftModel.from_pretrained(base, cfg.model.adapter_dir)

    logger.info("Merging adapter into base weights")
    merged = model.merge_and_unload()

    Path(cfg.model.merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(cfg.model.merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.adapter_dir)
    tokenizer.save_pretrained(cfg.model.merged_dir)
    logger.info("Merged model saved to %s", cfg.model.merged_dir)
    return cfg.model.merged_dir


class SQLPredictor:
    """Loads once, serves many. Used by the FastAPI app so the model isn't
    reloaded per request.
    """

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self._load()

    def _load(self) -> None:
        if self.cfg.serve.use_merged_weights and Path(self.cfg.model.merged_dir).exists():
            logger.info("Loading merged weights from %s", self.cfg.model.merged_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.merged_dir,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.merged_dir)
        else:
            logger.info(
                "Loading base model + adapter (%s)", self.cfg.model.adapter_dir
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.base_model_id,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.model = PeftModel.from_pretrained(base, self.cfg.model.adapter_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.adapter_dir)

        self.model.eval()
        self.model.config.use_cache = True

    @torch.no_grad()
    def predict(self, schema: str, question: str, max_new_tokens: Optional[int] = None) -> str:
        prompt = build_prompt(self.tokenizer, schema, question, self.cfg.data.system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.cfg.serve.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base weights")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = ProjectConfig.load(args.config)
    merge_and_save(cfg)
