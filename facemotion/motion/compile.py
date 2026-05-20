"""Compile structured facial motion states into image-edit prompts."""

from __future__ import annotations

import json
import math
import re
from typing import Pattern

from pydantic import ValidationError

from facemotion.motion.qwen import QwenChat
from facemotion.motion.schema import (
    EditPromptItem,
    EditPromptSet,
    MotionSpec,
    model_to_dict,
    validate_model,
)


COMPILE_SYSTEM_PROMPT = """
You are a prompt compiler for Qwen-Image-Edit-Lighting.

Your task is to convert a structured facial motion specification into self-contained portrait image editing prompts.

Output valid JSON only.
Do not output explanations, markdown, comments, or extra text.

Target model:
- Qwen-Image-Edit-Lighting
- The prompt must be written as an edit instruction for an existing portrait image.
- The model receives an input portrait and edits it.

Main editing goal:
- Preserve the original portrait identity and scene.
- Only change facial expression, gaze direction, and slight head pose according to the structured state.
- Do not change identity, age, hairstyle, clothing, skin tone, background, lighting, camera angle, camera framing, or image style.

Extremely important wording rules:
- The edit_prompt text must not contain these words or phrases:
  "frame", "keyframe", "start frame", "middle frame", "end frame",
  "first frame", "second frame", "third frame",
  "sequence", "grid", "collage", "split screen", "split-screen", "panel".
- Do not say "this is the start", "this is the middle", or "this is the end".
- The JSON may use state_index and temporal_progress for indexing, but the natural-language edit_prompt must not mention frame numbers or sequence position.
- The edit_prompt must describe a single edited portrait image only.
- Do not ask for multiple images.
- Do not use viewer-left or viewer-right.
- Use image-coordinate wording only:
  "toward the left side of the image"
  "toward the right side of the image"
  "straight forward"
  "slightly upward"
  "slightly downward"

Required content in every edit_prompt:
1. Begin with:
   "Edit this portrait while preserving the same identity, hairstyle, clothing, background, lighting, and camera framing."
2. State:
   "Only change the facial expression, gaze direction, and slight head pose."
3. Describe visible expression cues translated from AU values.
4. Describe gaze direction using image coordinates.
5. Describe head pose using image coordinates.
6. State:
   "The shoulders and upper body remain facing the camera directly."
7. State:
   "All key facial features are clearly visible."
8. Add preservation constraints:
   "Do not change the person's identity, age, hairstyle, skin tone, clothing, background, lighting, or camera framing."
9. Add anti-multi-image constraint:
   "Do not create multiple images."

State description rules:
- Each edit_prompt should describe one concrete visible portrait edit.
- Do not call any state a transition.
- Do not say "between the previous and final expression".
- Do not mention start, middle, end, or ordered position.
- Instead, describe the actual visible appearance:
  e.g. "The smile is subtly emerging..."
  e.g. "The smile is now clearly present but still gentle..."
  e.g. "The eyes have shifted partially toward the right side of the image..."
  e.g. "The eyes are clearly looking toward the right side of the image..."
  e.g. "The head remains mostly forward with only a very slight turn..."

AU translation guidance:
- AU1_inner_brow_raiser -> inner eyebrows slightly raised
- AU2_outer_brow_raiser -> outer eyebrows slightly raised
- AU4_brow_lowerer -> brows drawn slightly downward or together
- AU5_upper_lid_raiser -> eyes opened slightly wider
- AU6_cheek_raiser -> cheeks lifted, warmer smile, slight crow's-feet if natural
- AU7_lid_tightener -> eyelids slightly tightened or eyes gently narrowed
- AU9_nose_wrinkler -> subtle nose wrinkle
- AU10_upper_lip_raiser -> upper lip slightly raised
- AU12_lip_corner_puller -> mouth corners raised, smile
- AU15_lip_corner_depressor -> mouth corners lowered
- AU17_chin_raiser -> chin slightly raised or lower lip pushed upward
- AU20_lip_stretcher -> lips stretched horizontally
- AU23_lip_tightener -> lips pressed or tightened
- AU25_lips_part -> lips slightly parted
- AU26_jaw_drop -> jaw lowered or mouth open
- AU45_blink -> blink or eyes closed if high

Intensity wording:
- 0.00 to 0.10: neutral / barely visible / very subtle
- 0.10 to 0.30: slight / subtle
- 0.30 to 0.55: moderate / clearly visible but natural
- 0.55 to 0.80: strong but still realistic
- 0.80 to 1.00: intense / exaggerated only if explicitly needed

Gaze wording:
- gaze.yaw near 0: eyes looking straight forward
- gaze.yaw positive: eyes looking toward the right side of the image
- gaze.yaw negative: eyes looking toward the left side of the image
- gaze.pitch positive: eyes looking slightly upward
- gaze.pitch negative: eyes looking slightly downward

Head pose wording:
- head_pose.yaw near 0: head facing the camera directly
- positive yaw: head turned slightly toward the right side of the image
- negative yaw: head turned slightly toward the left side of the image
- pitch positive: chin slightly raised
- pitch negative: chin slightly lowered
- roll positive: head tilted slightly clockwise in image coordinates
- roll negative: head tilted slightly counterclockwise in image coordinates

Negative prompt requirements:
Each negative_prompt must be concise and must include:
- different identity
- changed hairstyle
- changed clothing
- changed background
- changed lighting
- exaggerated expression
- face turned into profile
- hidden eyes
- unclear facial features
- multiple images
- collage
- grid
- split-screen
- text
- watermark

Use this exact JSON schema:

{
  "task_type": "motion_spec_to_edit_prompts",
  "input_text": "...",
  "prompts": [
    {
      "state_index": 0,
      "temporal_progress": 0.0,
      "edit_prompt": "",
      "negative_prompt": ""
    }
  ]
}
"""

COMPILE_USER_PROMPT_TEMPLATE = """
Convert the following structured facial motion specification into Qwen-Image-Edit-Lighting edit prompts.

Original motion text:
"{input_text}"

Structured specification:
{structured_json}

Additional requirements:
1. Generate exactly one edit_prompt for each state in the structured specification.
2. Preserve state_index and temporal_progress from the structured specification.
3. The edit_prompt must be directly usable by an image edit model.
4. The edit_prompt must describe editing a single existing portrait image.
5. The edit_prompt must not contain the words or phrases:
   frame, keyframe, first frame, second frame, third frame, start frame, middle frame, end frame, sequence, grid, collage, panel, split screen, split-screen.
6. Preserve the same identity, hairstyle, clothing, background, lighting, and camera framing.
7. Only change facial expression, gaze direction, and slight head pose.
8. The shoulders and upper body remain facing the camera directly.
9. All key facial features are clearly visible.
10. Translate AU values into visible facial cues.
11. Translate gaze values into image-coordinate gaze descriptions.
12. Translate head_pose values into image-coordinate head orientation descriptions.
13. Do not use viewer-left or viewer-right.
14. Use "toward the left side of the image" or "toward the right side of the image" for direction.
15. Each item must be written as a concrete visible facial state, not as a transition between two states.
16. Include a concise negative_prompt for each item.
17. Return JSON only.
"""


FORBIDDEN_EDIT_PROMPT_PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("first frame", re.compile(r"\bfirst\s+frame\b", re.IGNORECASE)),
    ("second frame", re.compile(r"\bsecond\s+frame\b", re.IGNORECASE)),
    ("third frame", re.compile(r"\bthird\s+frame\b", re.IGNORECASE)),
    ("start frame", re.compile(r"\bstart\s+frame\b", re.IGNORECASE)),
    ("middle frame", re.compile(r"\bmiddle\s+frame\b", re.IGNORECASE)),
    ("end frame", re.compile(r"\bend\s+frame\b", re.IGNORECASE)),
    ("keyframe", re.compile(r"\bkeyframe\b", re.IGNORECASE)),
    ("frame", re.compile(r"(?<![A-Za-z])frame(?![A-Za-z])", re.IGNORECASE)),
    ("sequence", re.compile(r"\bsequence\b", re.IGNORECASE)),
    ("grid", re.compile(r"\bgrid\b", re.IGNORECASE)),
    ("collage", re.compile(r"\bcollage\b", re.IGNORECASE)),
    ("panel", re.compile(r"\bpanel\b", re.IGNORECASE)),
    ("split screen", re.compile(r"\bsplit\s+screen\b", re.IGNORECASE)),
    ("split-screen", re.compile(r"\bsplit-screen\b", re.IGNORECASE)),
    ("viewer", re.compile(r"\bviewer\b", re.IGNORECASE)),
]

REQUIRED_EDIT_PROMPT_SNIPPETS = [
    "Edit this portrait",
    "preserving the same identity",
    "Only change the facial expression, gaze direction, and slight head pose",
    "The shoulders and upper body remain facing the camera directly",
    "All key facial features are clearly visible",
    "Do not create multiple images",
]

REQUIRED_NEGATIVE_PROMPT_TERMS = [
    "different identity",
    "changed hairstyle",
    "changed clothing",
    "changed background",
    "changed lighting",
    "exaggerated expression",
    "face turned into profile",
    "hidden eyes",
    "unclear facial features",
    "multiple images",
    "collage",
    "grid",
    "split-screen",
    "text",
    "watermark",
]


class PromptCompiler:
    """Compile a validated MotionSpec into validated image-edit prompts."""

    def __init__(self, llm: QwenChat):
        self.llm = llm

    def run(self, spec: MotionSpec) -> EditPromptSet:
        structured_json = json.dumps(model_to_dict(spec), ensure_ascii=False, indent=2)
        user_prompt = COMPILE_USER_PROMPT_TEMPLATE.format(
            input_text=spec.input_text,
            structured_json=structured_json,
        )
        raw = self.llm.generate_json(
            system_prompt=COMPILE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_new_tokens=4096,
        )
        try:
            prompt_set = validate_model(EditPromptSet, raw)
            validate_prompt_set_against_spec(spec, prompt_set)
            return prompt_set
        except (ValidationError, ValueError) as exc:
            raw_output = getattr(self.llm, "last_raw_output", None)
            if raw_output is None:
                raw_output = json.dumps(raw, ensure_ascii=False)
            raise ValueError(
                "Failed to validate EditPromptSet from Qwen output: "
                f"{exc}\nRaw output:\n{raw_output}"
            ) from exc


def validate_prompt_set_against_spec(spec: MotionSpec, prompt_set: EditPromptSet) -> None:
    """Validate prompt count, indexing, progress values, and safe wording."""

    if prompt_set.input_text != spec.input_text:
        raise ValueError("EditPromptSet.input_text must match MotionSpec.input_text")

    if len(prompt_set.prompts) != len(spec.states):
        raise ValueError(
            "prompt count must match MotionSpec.states count: "
            f"{len(prompt_set.prompts)} != {len(spec.states)}"
        )

    previous_progress = -1.0
    for expected_index, (state, prompt) in enumerate(zip(spec.states, prompt_set.prompts)):
        if state.state_index != expected_index:
            raise ValueError(f"states must start at 0 and increase by 1; bad index {expected_index}")
        if prompt.state_index != expected_index:
            raise ValueError(
                f"state_index {prompt.state_index}: prompt index must be {expected_index}"
            )
        if prompt.state_index != state.state_index:
            raise ValueError(
                f"state_index {prompt.state_index}: prompt index does not match MotionState"
            )
        if not math.isclose(prompt.temporal_progress, state.temporal_progress, abs_tol=1e-6):
            raise ValueError(
                f"state_index {prompt.state_index}: temporal_progress "
                f"{prompt.temporal_progress} does not match MotionState {state.temporal_progress}"
            )
        if not 0.0 <= prompt.temporal_progress <= 1.0:
            raise ValueError(
                f"state_index {prompt.state_index}: temporal_progress must be in [0.0, 1.0]"
            )
        if prompt.temporal_progress <= previous_progress:
            raise ValueError(
                f"state_index {prompt.state_index}: temporal_progress must strictly increase"
            )
        previous_progress = prompt.temporal_progress
        validate_edit_prompt_text(prompt)
        validate_negative_prompt_text(prompt)

    if len(prompt_set.prompts) == 1:
        if not math.isclose(prompt_set.prompts[0].temporal_progress, 0.0, abs_tol=1e-6):
            raise ValueError("state_index 0: single prompt temporal_progress must be 0.0")
        return

    first = prompt_set.prompts[0]
    last = prompt_set.prompts[-1]
    if not math.isclose(first.temporal_progress, 0.0, abs_tol=1e-6):
        raise ValueError(f"state_index {first.state_index}: first temporal_progress must be 0.0")
    if not math.isclose(last.temporal_progress, 1.0, abs_tol=1e-6):
        raise ValueError(f"state_index {last.state_index}: last temporal_progress must be 1.0")


def validate_edit_prompt_text(prompt: EditPromptItem) -> None:
    """Validate one natural-language edit prompt."""

    for label, pattern in FORBIDDEN_EDIT_PROMPT_PATTERNS:
        if pattern.search(prompt.edit_prompt):
            raise ValueError(
                f"state_index {prompt.state_index}: edit_prompt contains forbidden "
                f"word or phrase {label!r}"
            )

    lowered = prompt.edit_prompt.lower()
    for snippet in REQUIRED_EDIT_PROMPT_SNIPPETS:
        if snippet.lower() not in lowered:
            raise ValueError(
                f"state_index {prompt.state_index}: edit_prompt must contain {snippet!r}"
            )


def validate_negative_prompt_text(prompt: EditPromptItem) -> None:
    """Validate one concise negative prompt."""

    lowered = prompt.negative_prompt.lower()
    for term in REQUIRED_NEGATIVE_PROMPT_TERMS:
        if term.lower() not in lowered:
            raise ValueError(
                f"state_index {prompt.state_index}: negative_prompt must include {term!r}"
            )
