"""End-to-end text-to-image-to-blendshape workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from facemotion.analyze.extract import BlendshapeBatchExtractor
from facemotion.analyze.mediapipe import MediaPipeBlendshapeExtractor
from facemotion.motion.pipeline import MotionPipeline
from facemotion.motion.qwen import DEFAULT_CHAT_MODEL, QwenChat
from facemotion.motion.schema import EditPromptSet, model_to_dict
from facemotion.render.editor import DEFAULT_EDIT_MODEL, QwenImageEditor
from facemotion.render.generate import StateRenderer


@dataclass(frozen=True)
class WorkflowResult:
    """Paths and in-memory payloads produced by the full workflow."""

    output_dir: str
    prompts_path: str
    states_dir: str
    blendshapes_path: str
    state_image_paths: list[str]
    prompts_payload: dict[str, Any]
    blendshapes: list[dict[str, Any]]


@dataclass(frozen=True)
class _PromptStageResult:
    public_payload: dict[str, Any]
    edit_prompts: EditPromptSet


class FacemotionBlendshapeWorkflow:
    """Generate prompts, render key-state images, and extract blendshapes."""

    def __init__(
        self,
        qwen_model: str = DEFAULT_CHAT_MODEL,
        edit_model: str = DEFAULT_EDIT_MODEL,
        gpu: int = 0,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        edit_steps: int = 4,
        width: int = 768,
        height: int = 768,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 1.0,
        use_lightning: bool = True,
        offload: str = "sequential",
    ):
        self.qwen_model = qwen_model
        self.edit_model = edit_model
        self.gpu = gpu
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.edit_steps = edit_steps
        self.width = width
        self.height = height
        self.true_cfg_scale = true_cfg_scale
        self.guidance_scale = guidance_scale
        self.use_lightning = use_lightning
        self.offload = offload

    def run(
        self,
        text: str,
        image_path: str,
        mediapipe_model_path: str,
        output_dir: str,
    ) -> WorkflowResult:
        """Run the full workflow and save prompts, images, and blendshapes."""

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        prompts_path = root / "prompts.json"
        states_dir = root / "states"
        blendshapes_path = root / "blendshapes.json"

        prompt_stage = self._generate_prompts(text)
        self._write_json(prompts_path, prompt_stage.public_payload)

        state_image_paths = self._render_states(
            image_path=image_path,
            prompt_set=prompt_stage.edit_prompts,
            output_dir=states_dir,
        )

        blendshapes = self._extract_blendshapes(
            states_dir=states_dir,
            mediapipe_model_path=mediapipe_model_path,
        )
        self._write_json(blendshapes_path, blendshapes)

        return WorkflowResult(
            output_dir=str(root),
            prompts_path=str(prompts_path),
            states_dir=str(states_dir),
            blendshapes_path=str(blendshapes_path),
            state_image_paths=state_image_paths,
            prompts_payload=prompt_stage.public_payload,
            blendshapes=blendshapes,
        )

    def _generate_prompts(self, text: str) -> _PromptStageResult:
        llm = QwenChat(
            model_id=self.qwen_model,
            gpu=self.gpu,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
        )
        motion_pipeline = MotionPipeline(llm)
        motion_spec, edit_prompts = motion_pipeline.run_with_spec(text)
        public_payload = {
            "motion_spec": model_to_dict(motion_spec),
            "edit_prompts": model_to_dict(edit_prompts),
        }
        return _PromptStageResult(public_payload=public_payload, edit_prompts=edit_prompts)

    def _render_states(
        self,
        image_path: str,
        prompt_set: EditPromptSet,
        output_dir: Path,
    ) -> list[str]:
        editor = QwenImageEditor(
            model_id=self.edit_model,
            gpu=self.gpu,
            steps=self.edit_steps,
            width=self.width,
            height=self.height,
            true_cfg_scale=self.true_cfg_scale,
            guidance_scale=self.guidance_scale,
            use_lightning=self.use_lightning,
            offload=self.offload,
        )
        renderer = StateRenderer(editor)
        try:
            return renderer.render(
                image_path=image_path,
                prompt_set=prompt_set,
                output_dir=str(output_dir),
            )
        finally:
            if hasattr(editor.model, "maybe_free_model_hooks"):
                editor.model.maybe_free_model_hooks()
            del editor

    def _extract_blendshapes(
        self,
        states_dir: Path,
        mediapipe_model_path: str,
    ) -> list[dict[str, Any]]:
        extractor = MediaPipeBlendshapeExtractor(model_path=mediapipe_model_path)
        batch = BlendshapeBatchExtractor(extractor)
        try:
            return batch.extract_dir(str(states_dir), pattern="*.png")
        finally:
            extractor.close()

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
