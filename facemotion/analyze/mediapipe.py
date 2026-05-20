"""MediaPipe Face Landmarker blendshape extraction.

Install the runtime dependency with:

    pip install mediapipe

Download a Face Landmarker task model, such as ``face_landmarker.task``, and
pass its path to ``MediaPipeBlendshapeExtractor`` or the CLI ``--model_path``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


BACKEND_NAME = "mediapipe_face_landmarker"


def compute_compact_metrics(blendshapes: dict[str, float]) -> dict[str, float]:
    """Compute compact expression metrics from MediaPipe blendshape scores."""

    def value(name: str) -> float:
        return float(blendshapes.get(name, 0.0))

    def mean(*names: str) -> float:
        return sum(value(name) for name in names) / len(names)

    return {
        "smile": mean("mouthSmileLeft", "mouthSmileRight"),
        "cheek": mean("cheekSquintLeft", "cheekSquintRight"),
        "blink": mean("eyeBlinkLeft", "eyeBlinkRight"),
        "squint": mean("eyeSquintLeft", "eyeSquintRight"),
        "eye_wide": mean("eyeWideLeft", "eyeWideRight"),
        "jaw_open": value("jawOpen"),
        "frown": mean("mouthFrownLeft", "mouthFrownRight"),
        "brow_raise": mean("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
        "brow_down": mean("browDownLeft", "browDownRight"),
        "mouth_pucker": value("mouthPucker"),
        "mouth_press": mean("mouthPressLeft", "mouthPressRight"),
    }


class MediaPipeBlendshapeExtractor:
    """Extract face blendshapes, landmarks, and face transforms from one image."""

    def __init__(
        self,
        model_path: str,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Face Landmarker model does not exist: {self.model_path}"
            )

        self.num_faces = num_faces
        self.min_face_detection_confidence = min_face_detection_confidence
        self.min_face_presence_confidence = min_face_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._mp: Any | None = None
        self._detector: Any | None = None

    def extract(self, image_path: str) -> dict[str, Any]:
        """Extract MediaPipe Face Landmarker results from a static image."""

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"input image does not exist: {path}")

        detector = self._get_detector()
        image = self._mp.Image.create_from_file(str(path))
        detection_result = detector.detect(image)

        face_landmarks = list(getattr(detection_result, "face_landmarks", None) or [])
        face_blendshapes = list(getattr(detection_result, "face_blendshapes", None) or [])
        face_transforms = list(
            getattr(detection_result, "facial_transformation_matrixes", None) or []
        )
        num_faces_detected = max(
            len(face_landmarks),
            len(face_blendshapes),
            len(face_transforms),
        )

        if num_faces_detected == 0:
            return {
                "image_path": str(path),
                "backend": BACKEND_NAME,
                "num_faces_detected": 0,
                "faces": [],
                "warning": "No face detected.",
            }

        faces: list[dict[str, Any]] = []
        for face_index in range(num_faces_detected):
            blendshapes = self._serialize_blendshapes(face_blendshapes, face_index)
            faces.append(
                {
                    "face_index": face_index,
                    "blendshapes": blendshapes,
                    "compact_metrics": compute_compact_metrics(blendshapes),
                    "landmarks": self._serialize_landmarks(face_landmarks, face_index),
                    "face_transform": self._serialize_face_transform(
                        face_transforms,
                        face_index,
                    ),
                }
            )

        return {
            "image_path": str(path),
            "backend": BACKEND_NAME,
            "num_faces_detected": num_faces_detected,
            "faces": faces,
        }

    def close(self) -> None:
        """Close the underlying MediaPipe detector if it has been created."""

        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()
        self._detector = None

    def __enter__(self) -> "MediaPipeBlendshapeExtractor":
        self._get_detector()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _get_detector(self) -> Any:
        if self._detector is None:
            mp, python, vision = _import_mediapipe_tasks()
            self._mp = mp
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=self.num_faces,
                min_face_detection_confidence=self.min_face_detection_confidence,
                min_face_presence_confidence=self.min_face_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
        return self._detector

    @staticmethod
    def _serialize_blendshapes(face_blendshapes: list[Any], face_index: int) -> dict[str, float]:
        if face_index >= len(face_blendshapes):
            return {}

        blendshapes: dict[str, float] = {}
        for category in face_blendshapes[face_index]:
            name = getattr(category, "category_name", None)
            if not name:
                continue
            blendshapes[str(name)] = float(getattr(category, "score", 0.0))
        return blendshapes

    @staticmethod
    def _serialize_landmarks(face_landmarks: list[Any], face_index: int) -> list[dict[str, float]]:
        if face_index >= len(face_landmarks):
            return []

        landmarks: list[dict[str, float]] = []
        for index, landmark in enumerate(face_landmarks[face_index]):
            landmarks.append(
                {
                    "index": index,
                    "x": float(getattr(landmark, "x", 0.0)),
                    "y": float(getattr(landmark, "y", 0.0)),
                    "z": float(getattr(landmark, "z", 0.0)),
                }
            )
        return landmarks

    @staticmethod
    def _serialize_face_transform(face_transforms: list[Any], face_index: int) -> list[list[float]] | None:
        if face_index >= len(face_transforms):
            return None

        matrix = face_transforms[face_index]
        if hasattr(matrix, "tolist"):
            matrix = matrix.tolist()

        if isinstance(matrix, (list, tuple)) and len(matrix) == 16:
            return [
                [float(matrix[row * 4 + col]) for col in range(4)]
                for row in range(4)
            ]

        if isinstance(matrix, (list, tuple)):
            rows: list[list[float]] = []
            for row in matrix:
                if not isinstance(row, (list, tuple)):
                    raise ValueError("facial transformation matrix must be 4x4 or length 16")
                rows.append([float(value) for value in row])
            return rows

        raise ValueError("unsupported facial transformation matrix format")


def _import_mediapipe_tasks() -> tuple[Any, Any, Any]:
    """Import the pip MediaPipe package even if a local source checkout exists."""

    project_root = Path(__file__).resolve().parents[2]
    local_checkout = project_root / "mediapipe"
    original_sys_path = list(sys.path)
    removed_local_module = False
    cached_modules: dict[str, Any] = {}

    existing = sys.modules.get("mediapipe")
    existing_file = Path(getattr(existing, "__file__", "") or "")
    if existing is not None and _is_relative_to(existing_file, local_checkout):
        removed_local_module = True
        for name in list(sys.modules):
            if name == "mediapipe" or name.startswith("mediapipe."):
                cached_modules[name] = sys.modules.pop(name)

    try:
        sys.path = [
            entry
            for entry in sys.path
            if not _sys_path_entry_points_to_local_mediapipe(entry, project_root, local_checkout)
        ]
        mp = importlib.import_module("mediapipe")
        python = importlib.import_module("mediapipe.tasks.python")
        vision = importlib.import_module("mediapipe.tasks.python.vision")
        return mp, python, vision
    except ImportError as exc:
        if removed_local_module:
            sys.modules.update(cached_modules)
        raise ImportError(
            "Failed to import the official MediaPipe Python package. Install it with "
            "`pip install mediapipe`. The local ./mediapipe source checkout is not a "
            "replacement for the pip wheel."
        ) from exc
    finally:
        sys.path = original_sys_path


def _sys_path_entry_points_to_local_mediapipe(
    entry: str,
    project_root: Path,
    local_checkout: Path,
) -> bool:
    resolved = Path(entry or ".").resolve()
    if resolved in {project_root, local_checkout}:
        return True
    return False


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
