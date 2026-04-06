from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis.plotting import plot_accuracy_comparison
from src.data.answer_extractor import normalize_answer
from src.data.preprocess import prepare_dataset
from src.infer.generator import GenerationConfig, batch_generate


@dataclass
class EvaluationConfig:
    model_path: str
    base_model_name_or_path: str | None
    dataset_name: str
    dataset_config: str
    split: str
    format_type: str
    max_samples: int | None
    batch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    output_dir: Path
    system_prompt: str
    seed: int = 42


def compute_exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def _save_jsonl(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def evaluate_model(config: EvaluationConfig) -> dict[str, float | int | str]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.split,
        sample_size=config.max_samples,
        format_type=config.format_type,
        system_prompt=config.system_prompt,
        seed=config.seed,
    )

    generation_config = GenerationConfig(
        model_path=config.model_path,
        base_model_name_or_path=config.base_model_name_or_path,
        format_type=config.format_type,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
        system_prompt=config.system_prompt,
    )

    predictions = batch_generate(dataset=dataset, config=generation_config)

    prediction_rows: list[dict] = []
    error_rows: list[dict] = []
    correct_count = 0

    for row, prediction in zip(dataset, predictions):
        is_correct = compute_exact_match(prediction["final_answer"], row["final_answer"])
        correct_count += int(is_correct)

        item = {
            "question": row["question"],
            "reference_cot": row["cot"],
            "reference_final_answer": row["final_answer"],
            "generated_text": prediction["generated_text"],
            "predicted_reasoning": prediction["reasoning"],
            "predicted_final_answer": prediction["final_answer"],
            "is_correct": is_correct,
        }
        prediction_rows.append(item)
        if not is_correct:
            error_rows.append(item)

    total = len(prediction_rows)
    accuracy = correct_count / total if total else 0.0
    metrics = {
        "model_path": config.model_path,
        "split": config.split,
        "num_examples": total,
        "num_correct": correct_count,
        "exact_match_accuracy": accuracy,
        "format_type": config.format_type,
    }

    _save_jsonl(prediction_rows, config.output_dir / "predictions.jsonl")
    _save_jsonl(error_rows, config.output_dir / "error_cases.jsonl")
    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, ensure_ascii=False)

    comparison_df = pd.DataFrame(
        [{"label": Path(config.model_path).name or config.model_path, "exact_match_accuracy": accuracy}]
    )
    plot_accuracy_comparison(comparison_df, config.output_dir / "accuracy_comparison.png")
    return metrics
