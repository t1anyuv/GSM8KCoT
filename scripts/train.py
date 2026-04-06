from __future__ import annotations

import argparse

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.train.trainer import load_yaml_config, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GSM8K CoT SFT model.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    artifacts = run_training(config)

    print("Training finished.")
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
