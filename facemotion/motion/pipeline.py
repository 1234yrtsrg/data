"""Two-stage facial motion prompt generation pipeline."""

from __future__ import annotations

from facemotion.motion.compile import PromptCompiler, validate_prompt_set_against_spec
from facemotion.motion.decompose import MotionDecomposer
from facemotion.motion.qwen import QwenChat
from facemotion.motion.schema import EditPromptSet, MotionSpec


class MotionPipeline:
    """Run motion decomposition followed by edit-prompt compilation."""

    def __init__(self, llm: QwenChat):
        self.decomposer = MotionDecomposer(llm)
        self.compiler = PromptCompiler(llm)

    def run(self, input_text: str) -> EditPromptSet:
        """Return only the compiled edit prompts."""

        _, prompt_set = self.run_with_spec(input_text)
        return prompt_set

    def run_with_spec(self, input_text: str) -> tuple[MotionSpec, EditPromptSet]:
        """Return both the structured MotionSpec and the compiled prompts."""

        motion_spec = self.decomposer.run(input_text)
        prompt_set = self.compiler.run(motion_spec)
        validate_prompt_set_against_spec(motion_spec, prompt_set)
        return motion_spec, prompt_set
