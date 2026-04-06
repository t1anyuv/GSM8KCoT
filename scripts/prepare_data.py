from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.preprocess import prepare_and_save_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSM8K data for CoT SFT.")
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--eval-sample-size", type=int, default=None)
    parser.add_argument(
        "--format-type",
        choices=["plain_cot", "chat_template"],
        default="plain_cot",
    )
    parser.add_argument("--save-format", choices=["jsonl", "parquet"], default="jsonl")
    parser.add_argument("--cache-dir", default="outputs/data")
    parser.add_argument(
        "--system-prompt",
        default="You are a careful math tutor. Solve the problem step by step and then give the final numeric answer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    outputs = prepare_and_save_splits(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        train_sample_size=args.sample_size,
        eval_sample_size=args.eval_sample_size,
        format_type=args.format_type,
        save_format=args.save_format,
        cache_dir=cache_dir,
        system_prompt=args.system_prompt,
        seed=args.seed,
    )

    print("Prepared datasets:")
    for split_name, path in outputs.items():
        print(f"  - {split_name}: {path}")


if __name__ == "__main__":
    main()
