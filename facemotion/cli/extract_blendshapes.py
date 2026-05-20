"""CLI for extracting MediaPipe Face Landmarker blendshapes.

Runtime dependency:

    pip install mediapipe

Download ``face_landmarker.task`` manually and pass it with ``--model_path``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from facemotion.analyze.extract import BlendshapeBatchExtractor
from facemotion.analyze.mediapipe import MediaPipeBlendshapeExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe face blendshapes from generated portrait states."
    )
    parser.add_argument("--image", help="Single image path.")
    parser.add_argument("--image_dir", help="Directory containing generated state images.")
    parser.add_argument("--pattern", default="*.png", help="Glob pattern for --image_dir.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the MediaPipe face_landmarker.task model.",
    )
    parser.add_argument("--output", required=True, help="Path to write JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image and not args.image_dir:
        raise ValueError("Provide at least one of --image or --image_dir")

    extractor = MediaPipeBlendshapeExtractor(model_path=args.model_path)
    batch_extractor = BlendshapeBatchExtractor(extractor)

    try:
        if args.image:
            payload: dict[str, Any] | list[dict[str, Any]] = extractor.extract(args.image)
        else:
            payload = batch_extractor.extract_dir(args.image_dir, pattern=args.pattern)
    finally:
        extractor.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote blendshape results to {output_path}")


if __name__ == "__main__":
    main()
