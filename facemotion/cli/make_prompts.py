"""CLI for generating facial motion specs and edit prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from facemotion.motion.decompose import EXAMPLE_TEXT
from facemotion.motion.pipeline import MotionPipeline
from facemotion.motion.qwen import DEFAULT_CHAT_MODEL, QwenChat
from facemotion.motion.schema import model_to_dict


def build_llm(args: argparse.Namespace) -> QwenChat:
    """Create the Qwen3-14B-Instruct chat adapter."""

    return QwenChat(
        model_id=args.model,
        gpu=args.gpu,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate facial motion edit prompts.")
    parser.add_argument(
        "--text",
        default=EXAMPLE_TEXT,
        help="Short facial motion description.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON output.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, for example: 0 or 7.")
    parser.add_argument(
        "--model",
        default=DEFAULT_CHAT_MODEL,
        help="Qwen3 instruct model id or local model path.",
    )
    parser.add_argument(
        "--device-map",
        choices=["cuda", "auto", "cpu"],
        default="cuda",
        help="Where to place the text model. Use 'auto' for accelerate device_map.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32", "auto"],
        default="bfloat16",
        help="Torch dtype for loading the text model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = build_llm(args)
    pipeline = MotionPipeline(llm)
    motion_spec, edit_prompts = pipeline.run_with_spec(args.text)

    payload: dict[str, Any] = {
        "motion_spec": model_to_dict(motion_spec),
        "edit_prompts": model_to_dict(edit_prompts),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote prompts to {output_path}")


if __name__ == "__main__":
    main()
