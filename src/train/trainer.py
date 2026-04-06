from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from src.analysis.plotting import plot_training_loss
from src.data.preprocess import prepare_dataset


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _parse_torch_dtype(value: str) -> str | torch.dtype:
    if value == "auto":
        return "auto"
    if not hasattr(torch, value):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return getattr(torch, value)


def _build_quantization_config(peft_config: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not peft_config.get("use_qlora", False):
        return None

    compute_dtype_name = peft_config.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = getattr(torch, compute_dtype_name)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=peft_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=peft_config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _build_lora_config(peft_config: dict[str, Any]) -> LoraConfig | None:
    if not peft_config.get("enabled", True):
        return None

    return LoraConfig(
        r=peft_config["r"],
        lora_alpha=peft_config["lora_alpha"],
        lora_dropout=peft_config["lora_dropout"],
        target_modules=peft_config["target_modules"],
        bias=peft_config.get("bias", "none"),
        task_type=peft_config.get("task_type", "CAUSAL_LM"),
    )


def _save_resolved_config(config: dict[str, Any], output_dir: Path) -> Path:
    config_path = output_dir / "resolved_config.json"
    with config_path.open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2, ensure_ascii=False)
    return config_path


def run_training(config: dict[str, Any]) -> dict[str, str]:
    seed = int(config.get("seed", 42))
    _set_all_seeds(seed)

    model_config = config["model"]
    data_config = config["data"]
    peft_config = config["peft"]
    training_config = config["training"]

    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = prepare_dataset(
        dataset_name=data_config["dataset_name"],
        dataset_config=data_config["dataset_config"],
        split=data_config.get("train_split", "train"),
        sample_size=data_config.get("train_sample_size"),
        format_type=data_config["format_type"],
        system_prompt=data_config.get("system_prompt"),
        seed=seed,
    )
    eval_dataset = prepare_dataset(
        dataset_name=data_config["dataset_name"],
        dataset_config=data_config["dataset_config"],
        split=data_config.get("eval_split", "test"),
        sample_size=data_config.get("eval_sample_size"),
        format_type=data_config["format_type"],
        system_prompt=data_config.get("system_prompt"),
        seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        use_fast=True,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = _build_quantization_config(peft_config)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        torch_dtype=_parse_torch_dtype(model_config.get("torch_dtype", "auto")),
        trust_remote_code=model_config.get("trust_remote_code", False),
        device_map="auto",
        quantization_config=quantization_config,
    )

    if training_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        run_name=training_config.get("run_name"),
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        logging_steps=training_config["logging_steps"],
        eval_strategy=training_config["eval_strategy"],
        eval_steps=training_config["eval_steps"],
        save_strategy=training_config["save_strategy"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        bf16=training_config["bf16"],
        fp16=training_config["fp16"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        report_to=training_config.get("report_to", "none"),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        seed=seed,
        dataset_text_field="text",
        max_length=data_config["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=_build_lora_config(peft_config),
    )

    trainer.train()

    final_checkpoint_dir = output_dir / "checkpoint-final"
    trainer.save_model(str(final_checkpoint_dir))
    tokenizer.save_pretrained(str(final_checkpoint_dir))

    metrics = trainer.evaluate()
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, ensure_ascii=False)

    loss_curve_path = plot_training_loss(trainer.state.log_history, output_dir / "loss_curve.png")
    resolved_config_path = _save_resolved_config(config, output_dir)

    return {
        "output_dir": str(output_dir),
        "final_checkpoint_dir": str(final_checkpoint_dir),
        "loss_curve_path": str(loss_curve_path),
        "resolved_config_path": str(resolved_config_path),
    }
