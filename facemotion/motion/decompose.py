"""Natural-language facial motion decomposition."""

from __future__ import annotations

import json

from pydantic import ValidationError

from facemotion.motion.qwen import QwenChat
from facemotion.motion.schema import MotionSpec, validate_model


EXAMPLE_TEXT = "A person is smiling gently and then glancing sideways."


DECOMPOSE_SYSTEM_PROMPT = """
You are a facial motion decomposition assistant.

Your task is to convert a short natural-language description of a human facial motion into a structured multi-state specification for controllable portrait image editing.

Output valid JSON only.
Do not output explanations, markdown, comments, or extra text.

Core task:
- Decompose the facial motion into a sequence of concrete key facial states.
- Do not force the output to exactly three states.
- Use as many states as needed to describe the motion clearly and naturally.
- For simple facial motions, usually generate 3 states.
- For very simple or nearly static descriptions, generate 1 or 2 states.
- For complex multi-step facial motions, generate 4 to 6 states.
- Each state must describe a concrete visible facial condition.
- Each state must be directly editable as a single portrait image.
- Do not create vague states that are only described as "transition".
- A state can be used if it represents a visually meaningful expression, gaze direction, or slight head-pose change.
- Adjacent states should be related but visually distinct.
- The full state sequence must be temporally coherent.

Semantic mapping:
- smile, frown, surprise, squint, eyebrow movement, mouth opening, etc. must be encoded mainly with action_units.
- glance, eye direction, eye contact, looking left/right/up/down must be encoded mainly with gaze.
- head turn, head tilt, nod, chin raise/lower must be encoded with head_pose.
- If the text implies that the eyes move more than the head, encode the change mainly in gaze, not head_pose.
- If "glance sideways" or similar is mentioned, prioritize gaze change over head turn.

Coordinate convention:
- All direction values must follow image coordinates.
- Positive gaze.yaw means the eyes look toward the right side of the image.
- Negative gaze.yaw means the eyes look toward the left side of the image.
- Positive head_pose.yaw means the face turns slightly toward the right side of the image.
- Negative head_pose.yaw means the face turns slightly toward the left side of the image.
- Do not use viewer-left or viewer-right concepts.

Value rules:
- action_units must use continuous values in [0.0, 1.0].
- head_pose values are degrees.
- gaze values are degrees.
- Keep head_pose subtle unless the input explicitly requires a head turn.
- Preserve realism.
- Avoid exaggerated expressions unless explicitly required.
- If a parameter is not mentioned, keep it neutral or minimally necessary.
- For subtle expressions, keep AU values moderate and natural.
- For a gentle smile, use mainly AU12_lip_corner_puller and slight AU6_cheek_raiser.
- Do not omit any AU key in any state.

Temporal rules:
- state_index must start from 0.
- state_index must increase by 1 for each state.
- temporal_progress must be a float in [0.0, 1.0].
- The first state must have temporal_progress 0.0.
- The last state must have temporal_progress 1.0.
- Intermediate states must have strictly increasing temporal_progress values.
- Do not restrict temporal_progress to only 0.0, 0.5, and 1.0.
- Motion should be monotonic or logically progressive when appropriate.
- Adjacent states should not be visually identical.
- Every intermediate state must be a concrete visible facial state, not an abstract transition label.

Global constraints:
- same_person must be true.
- same_scene must be true.
- same_lighting must be true.
- same_background must be true.
- shoulders_and_upper_body_facing_camera must be true.
- all_key_facial_features_clearly_visible must be true.
- The portrait type must be "head-and-shoulders realistic portrait".
- The style must be "natural realistic human face".

Use this exact JSON schema:

{
  "task_type": "facial_motion_spec",
  "input_text": "...",
  "direction_convention": "image_coordinates: positive yaw means toward the right side of the image, negative yaw means toward the left side of the image; never use viewer-left or viewer-right",
  "global_constraints": {
    "same_person": true,
    "same_scene": true,
    "same_lighting": true,
    "same_background": true,
    "shoulders_and_upper_body_facing_camera": true,
    "all_key_facial_features_clearly_visible": true,
    "portrait_type": "head-and-shoulders realistic portrait",
    "style": "natural realistic human face"
  },
  "states": [
    {
      "state_index": 0,
      "temporal_progress": 0.0,
      "expression": "",
      "action_units": {
        "AU1_inner_brow_raiser": 0.0,
        "AU2_outer_brow_raiser": 0.0,
        "AU4_brow_lowerer": 0.0,
        "AU5_upper_lid_raiser": 0.0,
        "AU6_cheek_raiser": 0.0,
        "AU7_lid_tightener": 0.0,
        "AU9_nose_wrinkler": 0.0,
        "AU10_upper_lip_raiser": 0.0,
        "AU12_lip_corner_puller": 0.0,
        "AU15_lip_corner_depressor": 0.0,
        "AU17_chin_raiser": 0.0,
        "AU20_lip_stretcher": 0.0,
        "AU23_lip_tightener": 0.0,
        "AU25_lips_part": 0.0,
        "AU26_jaw_drop": 0.0,
        "AU45_blink": 0.0
      },
      "head_pose": {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0
      },
      "gaze": {
        "yaw": 0.0,
        "pitch": 0.0
      },
      "visual_constraints": []
    }
  ]
}
"""

DECOMPOSE_USER_PROMPT_TEMPLATE = """
Convert the following facial motion text into a structured multi-state specification.

Input text:
"{input_text}"

Additional requirements:
1. Generate a natural number of key facial states based on the input motion.
2. Do not force exactly 3 states.
3. For simple motions, usually generate 3 states.
4. For very simple or nearly static descriptions, generate 1 or 2 states.
5. For complex multi-step motions, generate 4 to 6 states.
6. Each state must be a concrete visible facial condition that can be edited as a single portrait image.
7. Adjacent states must be visually different but temporally coherent.
8. Follow the required JSON schema exactly.
9. Do not omit any action unit key in any state.
10. Use realistic AU, gaze, and head pose values.
11. Keep the same identity, same scene, same lighting, same background.
12. The shoulders and upper body must remain facing the camera directly.
13. All key facial features must remain clearly visible.
14. If "glance sideways" is mentioned, prioritize gaze change over head turn.
15. If "gentle smile" is mentioned, use mainly AU12 and slight AU6.
16. Do not exaggerate head pose unless explicitly required.
17. Use image coordinates for all directions.
18. Never use viewer-left or viewer-right.
19. state_index must start at 0 and increase by 1.
20. temporal_progress must start at 0.0, end at 1.0, and strictly increase.
21. Return JSON only.
"""


class MotionDecomposer:
    """Convert a short text action description into a validated MotionSpec."""

    def __init__(self, llm: QwenChat):
        self.llm = llm

    def run(self, input_text: str) -> MotionSpec:
        user_prompt = DECOMPOSE_USER_PROMPT_TEMPLATE.format(input_text=input_text)
        raw = self.llm.generate_json(
            system_prompt=DECOMPOSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_new_tokens=4096,
        )
        try:
            return validate_model(MotionSpec, raw)
        except ValidationError as exc:
            raw_output = getattr(self.llm, "last_raw_output", None)
            if raw_output is None:
                raw_output = json.dumps(raw, ensure_ascii=False)
            raise ValueError(
                "Failed to validate MotionSpec from Qwen output: "
                f"{exc}\nRaw output:\n{raw_output}"
            ) from exc
