# GSM8K-CoT-SFT

An end-to-end open-source project for supervised fine-tuning (SFT) on GSM8K with chain-of-thought targets.

## Features

- Load GSM8K directly from Hugging Face
- Normalize each sample into `question / cot / final_answer / text`
- Support `plain_cot` and `chat_template`
- Cache processed data as `jsonl` or `parquet`
- Fine-tune small open models with LoRA or optional QLoRA
- Evaluate with exact match accuracy
- Save `predictions.jsonl`, `metrics.json`, `error_cases.jsonl`
- Plot training loss and accuracy comparison charts
- Launch a lightweight Gradio demo

## Project Structure

```text
GSM8KCoT/
├─ app/
│  └─ gradio_app.py
├─ configs/
│  └─ train_lora.yaml
├─ outputs/
│  ├─ data/
│  ├─ figures/
│  ├─ predictions/
│  └─ runs/
├─ scripts/
│  ├─ evaluate.py
│  ├─ prepare_data.py
│  ├─ run_experiments.py
│  ├─ summarize_results.py
│  └─ train.py
├─ src/
│  ├─ analysis/
│  │  ├─ __init__.py
│  │  └─ plotting.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ answer_extractor.py
│  │  └─ preprocess.py
│  ├─ eval/
│  │  ├─ __init__.py
│  │  └─ evaluator.py
│  ├─ infer/
│  │  ├─ __init__.py
│  │  └─ generator.py
│  └─ train/
│     ├─ __init__.py
│     └─ trainer.py
├─ requirements.txt
└─ README.md
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare data

```bash
python scripts/prepare_data.py --sample-size 500 --format-type plain_cot --save-format jsonl
```

### 2. Train

```bash
python scripts/train.py --config configs/train_lora.yaml
```

### 3. Evaluate

```bash
python scripts/evaluate.py --model-path outputs/runs/gsm8k-lora/checkpoint-final --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --format-type plain_cot --max-samples 200
```

### 4. Demo

```bash
python app/gradio_app.py --model-path outputs/runs/gsm8k-lora/checkpoint-final --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 5. Run experiment suite

```bash
python scripts/run_experiments.py --config configs/train_lora.yaml --studies baseline train_size prompt_format
```

### 6. Summarize results

```bash
python scripts/summarize_results.py --input-dir outputs --output-dir outputs/analysis/summary
```

## Data Format

Each processed record contains:

- `question`
- `cot`
- `final_answer`
- `text`

`final_answer` is extracted from the original GSM8K answer after the `####` delimiter.

## Training Formats

### `plain_cot`

```text
Question: ...

Reasoning:
...

Final Answer: ...
```

### `chat_template`

Chat-style formatting with a system prompt and user / assistant turns.

## Suggested Experiments

- Baseline vs fine-tuned model
- Train size comparison: `100 / 500 / 1000`
- `plain_cot` vs `chat_template`
- Different decoding parameters

## Outputs

- Training checkpoints under `outputs/runs/...`
- Processed data under `outputs/data/...`
- Evaluation artifacts under `outputs/predictions/...`
- Figures such as `loss_curve.png` and `accuracy_comparison.png`

## Reproducibility

- Fixed random seed
- YAML config
- Deterministic subset sampling
- Saved processed datasets
- Logged metrics and figures

## Notes

This initial version is an MVP for portfolio and GitHub showcase use. It prioritizes clean structure, reproducibility, and extensibility over benchmark-maximizing tricks.
