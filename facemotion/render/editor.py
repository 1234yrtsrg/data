"""Image-edit adapter for Qwen-Image-Edit-Lighting."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


LIGHTNING_LORA_ID = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_LORA_WEIGHT = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
DEFAULT_EDIT_MODEL = "Qwen/Qwen-Image-Edit-2511"

SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class QwenImageEditor:
    """Load or wrap Qwen-Image-Edit-2511 and save one edited image."""

    def __init__(
        self,
        model: Any | None = None,
        processor: Any | None = None,
        model_id: str = DEFAULT_EDIT_MODEL,
        gpu: int = 0,
        steps: int = 4,
        width: int = 768,
        height: int = 768,
        seed: int = 42,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 1.0,
        use_lightning: bool = True,
        offload: str = "sequential",
        **kwargs: Any,
    ):
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.gpu = gpu
        self.device = f"cuda:{gpu}"
        self.use_lightning = use_lightning
        self.offload = offload
        self.default_generation_kwargs = {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
            "seed": seed,
            "image_as_list": True,
        }
        self.default_generation_kwargs.update(kwargs)

    def edit(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str | None = None,
        output_path: str | None = None,
        **generation_kwargs: Any,
    ) -> str:
        """Edit one original portrait image and return the saved image path."""

        if self.model is None:
            self.model = self._load_pipeline()

        source_path = Path(image_path)
        if not source_path.exists():
            raise FileNotFoundError(f"input portrait image does not exist: {source_path}")

        if output_path is None:
            target_path = source_path.with_name(f"{source_path.stem}_edited.png")
        else:
            target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        merged_kwargs = dict(self.default_generation_kwargs)
        merged_kwargs.update(generation_kwargs)

        num_images = merged_kwargs.get("num_images_per_prompt", 1)
        if num_images != 1:
            raise ValueError("QwenImageEditor.edit must generate exactly one image")
        merged_kwargs["num_images_per_prompt"] = 1

        if hasattr(self.model, "edit"):
            result = self.model.edit(
                image_path=str(source_path),
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=str(target_path),
                **merged_kwargs,
            )
            return self._resolve_edit_result(result, target_path)

        if not callable(self.model):
            raise TypeError("QwenImageEditor model must be callable or provide edit()")

        pil_image = self._load_pil_image(source_path)
        image_as_list = bool(merged_kwargs.pop("image_as_list", False))
        image_input: Any = [pil_image] if image_as_list else pil_image

        seed = merged_kwargs.pop("seed", None)
        if seed is not None and "generator" not in merged_kwargs:
            merged_kwargs["generator"] = self._make_generator(int(seed))

        call_kwargs = {
            "image": image_input,
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt is not None else " ",
            **merged_kwargs,
        }

        result = self._call_model(call_kwargs)
        output_image = self._extract_single_image(result)
        output_image.save(target_path)
        return str(target_path)

    def _load_pipeline(self) -> Any:
        """Load the verified Qwen-Image-Edit-2511 Lightning pipeline."""

        try:
            import torch
            from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
        except ImportError as exc:
            raise ImportError(
                "QwenImageEditor requires torch and diffusers with "
                "QwenImageEditPlusPipeline support."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please run image editing on a server with an NVIDIA GPU."
            )

        device_count = torch.cuda.device_count()
        if self.gpu < 0 or self.gpu >= device_count:
            raise ValueError(
                f"Invalid GPU id {self.gpu}. This server has {device_count} CUDA device(s)."
            )

        torch.cuda.set_device(self.gpu)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )

        if self.use_lightning:
            try:
                pipeline.load_lora_weights(LIGHTNING_LORA_ID, weight_name=LIGHTNING_LORA_WEIGHT)
            except ValueError as exc:
                if "PEFT backend is required" in str(exc):
                    raise RuntimeError(
                        "Loading the Lightning LoRA requires the `peft` package. "
                        "Install it with: pip install peft"
                    ) from exc
                raise

        pipeline.set_progress_bar_config(disable=None)

        if self.offload == "none":
            pipeline.to(self.device)
        elif self.offload == "model":
            pipeline.enable_model_cpu_offload(gpu_id=self.gpu)
        elif self.offload == "sequential":
            pipeline.enable_sequential_cpu_offload(gpu_id=self.gpu)
        else:
            raise ValueError("offload must be one of: 'none', 'model', 'sequential'")

        if hasattr(pipeline, "vae"):
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()

        return pipeline

    @staticmethod
    def _load_pil_image(path: Path) -> Any:
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise ImportError("Pillow is required to load input portrait images") from exc
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")

    def _make_generator(self, seed: int) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to create a CUDA generator") from exc
        generator_device = self.device if torch.cuda.is_available() else "cpu"
        return torch.Generator(device=generator_device).manual_seed(seed)

    def _call_model(self, call_kwargs: dict[str, Any]) -> Any:
        try:
            import torch

            with torch.inference_mode():
                return self.model(**call_kwargs)
        except ImportError:
            return self.model(**call_kwargs)

    @staticmethod
    def _extract_single_image(result: Any) -> Any:
        if hasattr(result, "images"):
            images = result.images
        else:
            images = result

        if isinstance(images, (list, tuple)):
            if len(images) != 1:
                raise ValueError(f"edit model returned {len(images)} images; expected exactly one")
            return images[0]

        if hasattr(images, "save"):
            return images

        raise TypeError("edit model result must be a PIL image or an object with a single .images item")

    @staticmethod
    def _resolve_edit_result(result: Any, target_path: Path) -> str:
        if result is None:
            if not target_path.exists():
                raise FileNotFoundError(
                    f"edit wrapper returned None but did not create output file: {target_path}"
                )
            return str(target_path)

        if isinstance(result, (str, Path)):
            return str(result)

        output_image = QwenImageEditor._extract_single_image(result)
        output_image.save(target_path)
        return str(target_path)
