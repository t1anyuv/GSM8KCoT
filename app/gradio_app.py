from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.infer.generator import GenerationConfig, generate_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a GSM8K CoT demo with Gradio.")
    parser.add_argument("--model-path", required=True, help="Path to the model or adapter checkpoint.")
    parser.add_argument("--base-model", default=None, help="Optional base model path for LoRA adapters.")
    parser.add_argument("--format-type", choices=["plain_cot", "chat_template"], default="plain_cot")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--system-prompt",
        default="You are a careful math tutor. Solve the problem step by step and then give the final numeric answer.",
    )
    return parser.parse_args()


def build_demo(config: GenerationConfig) -> gr.Blocks:
    def _predict(question: str) -> tuple[str, str]:
        result = generate_one(question=question, config=config)
        return result["reasoning"], result["final_answer"]

    with gr.Blocks(title="GSM8K CoT SFT Demo") as demo:
        gr.Markdown("# GSM8K CoT SFT Demo")
        gr.Markdown("Enter a math word problem and get chain-of-thought reasoning plus the final answer.")

        question_box = gr.Textbox(
            label="Math Problem",
            lines=6,
            placeholder="If there are 12 apples and Tom eats 3, how many are left?",
        )
        submit_button = gr.Button("Solve")
        reasoning_box = gr.Textbox(label="Reasoning", lines=12)
        final_answer_box = gr.Textbox(label="Final Answer", lines=1)

        submit_button.click(
            fn=_predict,
            inputs=question_box,
            outputs=[reasoning_box, final_answer_box],
        )

    return demo


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        model_path=args.model_path,
        base_model_name_or_path=args.base_model,
        format_type=args.format_type,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        system_prompt=args.system_prompt,
    )
    demo = build_demo(config)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
