from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.answer_extractor import split_reasoning_and_answer


@dataclass
class GenerationConfig:
    model_path: str
    base_model_name_or_path: str | None = None
    format_type: str = "plain_cot"
    batch_size: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    system_prompt: str = (
        "You are a careful math tutor. Solve the problem step by step and then give the final numeric answer."
    )
    load_in_4bit: bool = False


def _resolve_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _build_generate_kwargs(config: GenerationConfig, pad_token_id: int) -> dict[str, int | float | bool]:
    generate_kwargs: dict[str, int | float | bool] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": pad_token_id,
    }
    if config.do_sample:
        generate_kwargs["temperature"] = config.temperature
        generate_kwargs["top_p"] = config.top_p
    return generate_kwargs


def load_model_and_tokenizer(
    model_path: str,
    base_model_name_or_path: str | None = None,
    load_in_4bit: bool = False,
):
    tokenizer_path = base_model_name_or_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=_resolve_dtype(),
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=_resolve_dtype(),
            quantization_config=quantization_config,
        )

    model.eval()
    return model, tokenizer


def build_inference_prompt(question: str, format_type: str, system_prompt: str) -> str:
    if format_type == "plain_cot":
        return f"Question: {question}\n\nReasoning:\n"

    if format_type == "chat_template":
        return (
            f"<|system|>\n{system_prompt}\n"
            "<|user|>\n"
            "Solve the following math word problem step by step. "
            "Return your reasoning and then the final answer.\n\n"
            f"Question: {question}\n"
            "<|assistant|>\n"
        )

    raise ValueError(f"Unsupported format_type: {format_type}")


def generate_one(question: str, config: GenerationConfig) -> dict[str, str]:
    model, tokenizer = load_model_and_tokenizer(
        model_path=config.model_path,
        base_model_name_or_path=config.base_model_name_or_path,
        load_in_4bit=config.load_in_4bit,
    )
    prompt = build_inference_prompt(question, config.format_type, config.system_prompt)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_kwargs = _build_generate_kwargs(config, tokenizer.pad_token_id)
    outputs = model.generate(
        **encoded,
        **generate_kwargs,
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
    reasoning, final_answer = split_reasoning_and_answer(generated_text)
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "reasoning": reasoning,
        "final_answer": final_answer,
    }


def batch_generate(dataset: Dataset, config: GenerationConfig) -> list[dict[str, str]]:
    model, tokenizer = load_model_and_tokenizer(
        model_path=config.model_path,
        base_model_name_or_path=config.base_model_name_or_path,
        load_in_4bit=config.load_in_4bit,
    )
    results: list[dict[str, str]] = []

    for start_idx in range(0, len(dataset), config.batch_size):
        batch = dataset[start_idx : start_idx + config.batch_size]
        prompts = [
            build_inference_prompt(question, config.format_type, config.system_prompt)
            for question in batch["question"]
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        generate_kwargs = _build_generate_kwargs(config, tokenizer.pad_token_id)
        outputs = model.generate(
            **encoded,
            **generate_kwargs,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for prompt, full_text in zip(prompts, decoded):
            generated_text = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
            reasoning, final_answer = split_reasoning_and_answer(generated_text)
            results.append(
                {
                    "generated_text": generated_text,
                    "reasoning": reasoning,
                    "final_answer": final_answer,
                }
            )

    return results
