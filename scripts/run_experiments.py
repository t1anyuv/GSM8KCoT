from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.analysis.plotting import plot_accuracy_comparison
from src.eval.evaluator import EvaluationConfig, evaluate_model
from src.train.trainer import load_yaml_config, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a suite of GSM8K CoT SFT experiments.")
    parser.add_argument("--config", required=True, help="Base training config path.")
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=["baseline", "train_size", "prompt_format", "decoding"],
        default=["baseline", "train_size", "prompt_format", "decoding"],
        help="Experiment groups to run.",
    )
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--prompt-formats", nargs="+", default=["plain_cot", "chat_template"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7])
    parser.add_argument("--top-ps", nargs="+", type=float, default=[1.0, 0.9])
    parser.add_argument("--max-new-tokens-list", nargs="+", type=int, default=[256, 384])
    parser.add_argument("--eval-max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/experiments/default")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _build_eval_config(
    config: dict[str, Any],
    model_path: str,
    output_dir: Path,
    format_type: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    do_sample: bool | None = None,
) -> EvaluationConfig:
    data_config = config["data"]
    generation_config = config.get("generation", {})

    resolved_temperature = generation_config.get("temperature", 0.0) if temperature is None else temperature
    resolved_top_p = generation_config.get("top_p", 1.0) if top_p is None else top_p
    resolved_max_new_tokens = (
        generation_config.get("max_new_tokens", 256) if max_new_tokens is None else max_new_tokens
    )
    resolved_do_sample = generation_config.get("do_sample", False) if do_sample is None else do_sample

    return EvaluationConfig(
        model_path=model_path,
        base_model_name_or_path=config["model"]["name_or_path"],
        dataset_name=data_config["dataset_name"],
        dataset_config=data_config["dataset_config"],
        split=data_config.get("eval_split", "test"),
        format_type=format_type,
        max_samples=data_config.get("eval_sample_size"),
        batch_size=config["training"]["per_device_eval_batch_size"],
        max_new_tokens=resolved_max_new_tokens,
        temperature=resolved_temperature,
        top_p=resolved_top_p,
        do_sample=resolved_do_sample,
        output_dir=output_dir,
        system_prompt=data_config.get("system_prompt"),
        seed=config.get("seed", 42),
    )


def _run_baseline(base_config: dict[str, Any], experiment_root: Path) -> dict[str, Any]:
    format_type = base_config["data"]["format_type"]
    output_dir = experiment_root / "baseline"
    eval_config = _build_eval_config(
        base_config,
        model_path=base_config["model"]["name_or_path"],
        output_dir=output_dir,
        format_type=format_type,
    )
    metrics = evaluate_model(eval_config)
    metrics["label"] = f"baseline-{format_type}"
    metrics["study"] = "baseline"
    return metrics


def _run_train_size_experiment(
    base_config: dict[str, Any],
    experiment_root: Path,
    train_size: int,
    skip_existing: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    format_type = config["data"]["format_type"]
    run_dir = experiment_root / "train_size" / f"size_{train_size}_{format_type}"
    config["data"]["train_sample_size"] = train_size
    config["training"]["output_dir"] = str(run_dir / "train")
    config["training"]["run_name"] = f"gsm8k-cot-size-{train_size}-{format_type}"

    checkpoint_dir = Path(config["training"]["output_dir"]) / "checkpoint-final"
    if not (skip_existing and checkpoint_dir.exists()):
        run_training(config)

    eval_config = _build_eval_config(
        config,
        model_path=str(checkpoint_dir),
        output_dir=run_dir / "eval",
        format_type=format_type,
    )
    metrics = evaluate_model(eval_config)
    metrics["label"] = f"ft-size-{train_size}-{format_type}"
    metrics["study"] = "train_size"
    metrics["train_sample_size"] = train_size
    return metrics


def _run_prompt_experiment(
    base_config: dict[str, Any],
    experiment_root: Path,
    format_type: str,
    skip_existing: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    train_size = config["data"].get("train_sample_size")
    run_dir = experiment_root / "prompt_format" / f"{format_type}"
    config["data"]["format_type"] = format_type
    config["training"]["output_dir"] = str(run_dir / "train")
    config["training"]["run_name"] = f"gsm8k-cot-format-{format_type}"

    checkpoint_dir = Path(config["training"]["output_dir"]) / "checkpoint-final"
    if not (skip_existing and checkpoint_dir.exists()):
        run_training(config)

    eval_config = _build_eval_config(
        config,
        model_path=str(checkpoint_dir),
        output_dir=run_dir / "eval",
        format_type=format_type,
    )
    metrics = evaluate_model(eval_config)
    metrics["label"] = f"ft-format-{format_type}"
    metrics["study"] = "prompt_format"
    metrics["train_sample_size"] = train_size
    return metrics


def _run_decoding_experiment(
    base_config: dict[str, Any],
    experiment_root: Path,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    format_type = base_config["data"]["format_type"]
    checkpoint_dir = Path(base_config["training"]["output_dir"]) / "checkpoint-final"
    if not checkpoint_dir.exists():
        run_training(base_config)

    do_sample = temperature > 0.0
    run_name = f"temp_{temperature}_top_p_{top_p}_tokens_{max_new_tokens}"
    eval_config = _build_eval_config(
        base_config,
        model_path=str(checkpoint_dir),
        output_dir=experiment_root / "decoding" / run_name,
        format_type=format_type,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    metrics = evaluate_model(eval_config)
    metrics["label"] = f"decode-{run_name}"
    metrics["study"] = "decoding"
    metrics["temperature"] = temperature
    metrics["top_p"] = top_p
    metrics["max_new_tokens"] = max_new_tokens
    return metrics


def _save_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not results:
        return

    df = pd.DataFrame(results).sort_values(by=["study", "label"]).reset_index(drop=True)
    df.to_csv(output_dir / "experiment_summary.csv", index=False)
    df.to_json(output_dir / "experiment_summary.json", orient="records", indent=2, force_ascii=False)
    plot_accuracy_comparison(df[["label", "exact_match_accuracy"]], output_dir / "experiment_comparison.png")


def main() -> None:
    args = parse_args()
    base_config = load_yaml_config(args.config)
    base_config["data"]["eval_sample_size"] = args.eval_max_samples
    base_config["training"]["per_device_eval_batch_size"] = args.batch_size
    experiment_root = Path(args.output_dir)

    results: list[dict[str, Any]] = []

    if "baseline" in args.studies:
        results.append(_run_baseline(base_config, experiment_root))

    if "train_size" in args.studies:
        for train_size in args.train_sizes:
            results.append(
                _run_train_size_experiment(
                    base_config=base_config,
                    experiment_root=experiment_root,
                    train_size=train_size,
                    skip_existing=args.skip_existing,
                )
            )

    if "prompt_format" in args.studies:
        for format_type in args.prompt_formats:
            results.append(
                _run_prompt_experiment(
                    base_config=base_config,
                    experiment_root=experiment_root,
                    format_type=format_type,
                    skip_existing=args.skip_existing,
                )
            )

    if "decoding" in args.studies:
        for temperature in args.temperatures:
            for top_p in args.top_ps:
                for max_new_tokens in args.max_new_tokens_list:
                    results.append(
                        _run_decoding_experiment(
                            base_config=copy.deepcopy(base_config),
                            experiment_root=experiment_root,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_new_tokens,
                        )
                    )

    _save_summary(results, experiment_root)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
