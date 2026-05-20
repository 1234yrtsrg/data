"""CLI for rendering facial motion state images from prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from facemotion.motion.schema import EditPromptSet, validate_model
from facemotion.render.editor import DEFAULT_EDIT_MODEL, QwenImageEditor
from facemotion.render.generate import StateRenderer


def build_editor(args: argparse.Namespace) -> QwenImageEditor:
    """Create the Qwen-Image-Edit-Lighting editor adapter."""

    return QwenImageEditor(
        model_id=args.model,
        gpu=args.gpu,
        steps=args.steps,
        width=args.width,
        height=args.height,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        use_lightning=not args.no_lightning,
        offload=args.offload,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render facial motion state images.")
    parser.add_argument("--image", required=True, help="Input portrait image path.")
    parser.add_argument("--prompts", required=True, help="JSON file from make_prompts.")
    parser.add_argument("--output_dir", required=True, help="Directory for rendered states.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, for example: 0 or 7.")
    parser.add_argument(
        "--model",
        default=DEFAULT_EDIT_MODEL,
        help="Qwen image edit model id or local model path.",
    )
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps.")
    parser.add_argument("--width", type=int, default=768, help="Output width.")
    parser.add_argument("--height", type=int, default=768, help="Output height.")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0, help="True CFG scale.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument("--no-lightning", action="store_true", help="Disable Lightning LoRA.")
    parser.add_argument(
        "--offload",
        choices=["model", "sequential", "none"],
        default="sequential",
        help=(
            "Memory strategy. Use 'none' for speed on large GPUs, 'model' for medium "
            "memory, and 'sequential' for lower memory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts_path = Path(args.prompts)
    payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    if "edit_prompts" not in payload:
        raise ValueError(f"{prompts_path} must contain an 'edit_prompts' object")

    prompt_set = validate_model(EditPromptSet, payload["edit_prompts"])
    editor = build_editor(args)
    renderer = StateRenderer(editor)
    output_paths = renderer.render(
        image_path=args.image,
        prompt_set=prompt_set,
        output_dir=args.output_dir,
    )

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
