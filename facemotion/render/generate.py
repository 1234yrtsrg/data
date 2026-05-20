"""Render facial motion states from compiled edit prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from facemotion.motion.schema import EditPromptSet
from facemotion.render.editor import QwenImageEditor


class StateRenderer:
    """Generate one edited portrait image per key facial state."""

    def __init__(self, editor: QwenImageEditor):
        self.editor = editor

    def render(
        self,
        image_path: str,
        prompt_set: EditPromptSet,
        output_dir: str,
        **generation_kwargs: Any,
    ) -> list[str]:
        """Render each state using the same original portrait image."""

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        output_paths: list[str] = []
        for item in sorted(prompt_set.prompts, key=lambda prompt: prompt.state_index):
            output_path = target_dir / f"{item.state_index:03d}.png"
            rendered_path = self.editor.edit(
                image_path=image_path,
                prompt=item.edit_prompt,
                negative_prompt=item.negative_prompt,
                output_path=str(output_path),
                **generation_kwargs,
            )
            output_paths.append(rendered_path)
        return output_paths
