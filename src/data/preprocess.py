from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from src.data.answer_extractor import extract_final_answer_from_reference


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step and then give the final numeric answer."
)


def load_gsm8k_split(
    dataset_name: str = "gsm8k",
    dataset_config: str = "main",
    split: str = "train",
) -> Dataset:
    return load_dataset(dataset_name, dataset_config, split=split)


def sample_dataset(dataset: Dataset, sample_size: int | None, seed: int) -> Dataset:
    if sample_size is None or sample_size >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(sample_size))


def standardize_example(example: dict[str, Any]) -> dict[str, str]:
    cot, final_answer = extract_final_answer_from_reference(example["answer"])
    return {
        "question": example["question"].strip(),
        "cot": cot,
        "final_answer": final_answer,
    }


def build_training_text(
    example: dict[str, str],
    format_type: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    if format_type == "plain_cot":
        return (
            f"Question: {example['question']}\n\n"
            f"Reasoning:\n{example['cot']}\n\n"
            f"Final Answer: {example['final_answer']}"
        )

    if format_type == "chat_template":
        user_message = (
            "Solve the following math word problem step by step. "
            "Return your reasoning and then the final answer.\n\n"
            f"Question: {example['question']}"
        )
        assistant_message = f"{example['cot']}\n\n#### {example['final_answer']}"
        return (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{user_message}\n"
            f"<|assistant|>\n{assistant_message}"
        )

    raise ValueError(f"Unsupported format_type: {format_type}")


def prepare_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    sample_size: int | None,
    format_type: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    seed: int = 42,
) -> Dataset:
    dataset = load_gsm8k_split(dataset_name=dataset_name, dataset_config=dataset_config, split=split)
    dataset = sample_dataset(dataset, sample_size=sample_size, seed=seed)

    def _map_fn(example: dict[str, Any]) -> dict[str, str]:
        standardized = standardize_example(example)
        standardized["text"] = build_training_text(
            standardized,
            format_type=format_type,
            system_prompt=system_prompt,
        )
        return standardized

    return dataset.map(
        _map_fn,
        remove_columns=dataset.column_names,
        desc=f"Preparing {split} split",
    )


def save_dataset(dataset: Dataset, output_path: str | Path, save_format: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as file_obj:
            for row in dataset:
                file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
        return output_path

    if save_format == "parquet":
        dataset.to_parquet(str(output_path))
        return output_path

    raise ValueError(f"Unsupported save_format: {save_format}")


def build_cache_filename(split: str, format_type: str, sample_size: int | None, save_format: str) -> str:
    size_part = "full" if sample_size is None else str(sample_size)
    return f"gsm8k_{split}_{format_type}_{size_part}.{save_format}"


def prepare_and_save_splits(
    dataset_name: str,
    dataset_config: str,
    train_split: str,
    eval_split: str,
    train_sample_size: int | None,
    eval_sample_size: int | None,
    format_type: str,
    save_format: str,
    cache_dir: str | Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    seed: int = 42,
) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    outputs: dict[str, Path] = {}

    for split_name, sample_size in ((train_split, train_sample_size), (eval_split, eval_sample_size)):
        prepared = prepare_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split_name,
            sample_size=sample_size,
            format_type=format_type,
            system_prompt=system_prompt,
            seed=seed,
        )
        filename = build_cache_filename(split_name, format_type, sample_size, save_format)
        outputs[split_name] = save_dataset(prepared, cache_dir / filename, save_format)

    return outputs
