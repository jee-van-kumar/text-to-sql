"""Fine-tune TinyLlama-1.1B with LoRA on the text-to-SQL task.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --train.num_train_epochs 3
"""
from __future__ import annotations

import argparse
import logging
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import ProjectConfig
from .data import load_and_prepare, tokenize_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SQL-LoRA model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--report-to",
        type=str,
        default=None,
        help='Override train.report_to, e.g. "wandb"',
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs")
    return parser.parse_args()


def load_base_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig.load(args.config)

    if args.report_to:
        cfg.train.report_to = args.report_to
    if args.epochs:
        cfg.train.num_train_epochs = args.epochs

    logger.info("Loading base model: %s", cfg.model.base_model_id)
    model, tokenizer = load_base_model(cfg.model.base_model_id)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Base model parameters: %.1fM", n_params / 1e6)

    logger.info("Loading and formatting dataset: %s", cfg.data.dataset_id)
    train_ds, eval_ds = load_and_prepare(tokenizer, cfg.data)
    logger.info("Train size=%d  Eval size=%d", len(train_ds), len(eval_ds))

    train_tok = tokenize_dataset(train_ds, tokenizer, cfg.model.max_seq_length)
    eval_tok = tokenize_dataset(eval_ds, tokenizer, cfg.model.max_seq_length)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
        target_modules=cfg.lora.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if cfg.train.report_to == "wandb":
        import wandb

        wandb.init(
            project="sql-lora",
            name=cfg.train.run_name or "tinyllama-sql-lora",
            config=cfg.to_dict(),
        )

    sft_config = SFTConfig(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_steps=cfg.train.warmup_steps,
        optim="adamw_torch",
        fp16=cfg.train.fp16 and torch.cuda.is_available(),
        bf16=cfg.train.bf16,
        logging_steps=cfg.train.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_strategy="epoch",
        report_to=cfg.train.report_to,
        seed=cfg.train.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    if torch.cuda.is_available():
        logger.info(
            "Peak GPU memory: %.2f GB", torch.cuda.max_memory_allocated() / 1024**3
        )

    logger.info("Saving adapter to %s", cfg.model.adapter_dir)
    model.save_pretrained(cfg.model.adapter_dir)
    tokenizer.save_pretrained(cfg.model.adapter_dir)
    cfg.save(f"{cfg.model.adapter_dir}/run_config.yaml")

    logger.info("Done.")


if __name__ == "__main__":
    sys.exit(main() or 0)
