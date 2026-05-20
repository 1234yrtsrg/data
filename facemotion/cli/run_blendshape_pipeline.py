"""CLI for the full text-to-blendshapes facemotion workflow."""

from __future__ import annotations

import argparse

from facemotion.motion.decompose import EXAMPLE_TEXT
from facemotion.motion.qwen import DEFAULT_CHAT_MODEL
from facemotion.render.editor import DEFAULT_EDIT_MODEL
from facemotion.workflow.blendshapes import FacemotionBlendshapeWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text -> prompts -> edited state images -> MediaPipe blendshapes."
    )
    parser.add_argument(
        "--text",
        default=EXAMPLE_TEXT,
        help="Short facial motion description.",
    )
    parser.add_argument("--image", required=True, help="Input portrait image path.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the MediaPipe face_landmarker.task model.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory that will contain prompts.json, states/, and blendshapes.json.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_CHAT_MODEL,
        help="Qwen3 model id or local model path.",
    )
    parser.add_argument(
        "--edit-model",
        default=DEFAULT_EDIT_MODEL,
        help="Qwen image edit model id or local model path.",
    )
    parser.add_argument(
        "--device-map",
        choices=["cuda", "auto", "cpu"],
        default="auto",
        help="Where to place the text model. Use 'auto' for V100 32GB or limited memory.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32", "auto"],
        default="float16",
        help="Torch dtype for the text model. V100 should use float16.",
    )
    parser.add_argument("--steps", type=int, default=4, help="Image edit inference steps.")
    parser.add_argument("--width", type=int, default=768, help="Output image width.")
    parser.add_argument("--height", type=int, default=768, help="Output image height.")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0, help="Image edit true CFG.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Image edit guidance.")
    parser.add_argument("--no-lightning", action="store_true", help="Disable Lightning LoRA.")
    parser.add_argument(
        "--offload",
        choices=["model", "sequential", "none"],
        default="sequential",
        help="Image edit memory strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = FacemotionBlendshapeWorkflow(
        qwen_model=args.qwen_model,
        edit_model=args.edit_model,
        gpu=args.gpu,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        edit_steps=args.steps,
        width=args.width,
        height=args.height,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        use_lightning=not args.no_lightning,
        offload=args.offload,
    )
    result = workflow.run(
        text=args.text,
        image_path=args.image,
        mediapipe_model_path=args.model_path,
        output_dir=args.output_dir,
    )
    print(f"Wrote prompts to {result.prompts_path}")
    print(f"Wrote state images to {result.states_dir}")
    print(f"Wrote blendshapes to {result.blendshapes_path}")


if __name__ == "__main__":
    main()
