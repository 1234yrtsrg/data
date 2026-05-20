"""Batch extraction helpers for generated facial state images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from facemotion.analyze.mediapipe import BACKEND_NAME, MediaPipeBlendshapeExtractor


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class BlendshapeBatchExtractor:
    """Run a MediaPipe blendshape extractor over images or a directory."""

    def __init__(self, extractor: MediaPipeBlendshapeExtractor):
        self.extractor = extractor

    def extract_images(self, image_paths: list[str]) -> list[dict[str, Any]]:
        """Extract image results in the same order as ``image_paths``."""

        results: list[dict[str, Any]] = []
        for image_path in image_paths:
            try:
                results.append(self.extractor.extract(image_path))
            except Exception as exc:
                results.append(
                    {
                        "image_path": image_path,
                        "backend": BACKEND_NAME,
                        "error": str(exc),
                    }
                )
        return results

    def extract_dir(
        self,
        image_dir: str,
        pattern: str = "*.png",
    ) -> list[dict[str, Any]]:
        """Extract all supported images matching ``pattern``, sorted by filename."""

        directory = Path(image_dir)
        if not directory.exists():
            raise FileNotFoundError(f"image directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"image_dir must be a directory: {directory}")

        image_paths = [
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        image_paths.sort(key=lambda path: path.name)
        return self.extract_images([str(path) for path in image_paths])
