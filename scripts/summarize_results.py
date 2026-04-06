from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.analysis.plotting import plot_accuracy_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics from experiment folders.")
    parser.add_argument("--input-dir", default="outputs", help="Root directory to scan for metrics.json files.")
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/summary",
        help="Directory for aggregated CSV, JSON, and comparison plot.",
    )
    return parser.parse_args()


def collect_metrics(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in input_dir.rglob("metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as file_obj:
            metrics = json.load(file_obj)
        metrics["metrics_path"] = str(metrics_path)
        metrics["label"] = metrics.get("label") or metrics_path.parent.name
        rows.append(metrics)
    return rows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_metrics(input_dir)
    if not rows:
        print(f"No metrics.json files found under: {input_dir}")
        return

    df = pd.DataFrame(rows).sort_values(by=["exact_match_accuracy", "label"], ascending=[False, True])
    df.to_csv(output_dir / "all_metrics.csv", index=False)
    df.to_json(output_dir / "all_metrics.json", orient="records", indent=2, force_ascii=False)
    plot_accuracy_comparison(df[["label", "exact_match_accuracy"]], output_dir / "all_metrics.png")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
