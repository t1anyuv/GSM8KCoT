from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.eval.evaluator import EvaluationConfig, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a base or fine-tuned model on GSM8K.")
    parser.add_argument("--model-path", required=True, help="Path to model or adapter checkpoint.")
    parser.add_argument("--base-model", default=None, help="Base model path for loading LoRA adapters.")
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--format-type", choices=["plain_cot", "chat_template"], default="plain_cot")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--output-dir", default="outputs/predictions/latest_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system-prompt",
        default="You are a careful math tutor. Solve the problem step by step and then give the final numeric answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        model_path=args.model_path,
        base_model_name_or_path=args.base_model,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        format_type=args.format_type,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        output_dir=Path(args.output_dir),
        system_prompt=args.system_prompt,
        seed=args.seed,
    )
    metrics = evaluate_model(config)

    print("Evaluation finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
