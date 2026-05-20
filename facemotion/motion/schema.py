"""Pydantic schemas for facial motion specifications and edit prompts."""

from __future__ import annotations

import math
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


T = TypeVar("T", bound=BaseModel)


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Return a plain dict for Pydantic v2 models."""

    return model.model_dump()


def validate_model(model_type: type[T], data: Any) -> T:
    """Validate data using a Pydantic model class."""

    return model_type.model_validate(data)


def _is_close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-6)


def _validate_indexed_progression(items: list[Any], item_name: str) -> None:
    if not items:
        raise ValueError(f"{item_name} must contain at least one item")

    previous_progress = -1.0
    for expected_index, item in enumerate(items):
        if item.state_index != expected_index:
            raise ValueError(
                f"{item_name}[{expected_index}] must have state_index "
                f"{expected_index}, got {item.state_index}"
            )
        if not 0.0 <= item.temporal_progress <= 1.0:
            raise ValueError(
                f"{item_name}[{expected_index}] temporal_progress must be in [0.0, 1.0]"
            )
        if item.temporal_progress <= previous_progress:
            raise ValueError(f"{item_name} temporal_progress values must strictly increase")
        previous_progress = item.temporal_progress

    if len(items) == 1:
        # A one-state static description has no duration; keep it anchored at 0.0.
        if not _is_close(items[0].temporal_progress, 0.0):
            raise ValueError(f"single {item_name} item must have temporal_progress 0.0")
        return

    if not _is_close(items[0].temporal_progress, 0.0):
        raise ValueError(f"first {item_name} item must have temporal_progress 0.0")
    if not _is_close(items[-1].temporal_progress, 1.0):
        raise ValueError(f"last {item_name} item must have temporal_progress 1.0")


def _states_are_visually_identical(left: "MotionState", right: "MotionState") -> bool:
    if left.expression.strip().lower() != right.expression.strip().lower():
        return False

    left_au = model_to_dict(left.action_units)
    right_au = model_to_dict(right.action_units)
    for key, left_value in left_au.items():
        if not _is_close(left_value, right_au[key]):
            return False

    for attr in ("yaw", "pitch", "roll"):
        if not _is_close(getattr(left.head_pose, attr), getattr(right.head_pose, attr)):
            return False
    for attr in ("yaw", "pitch"):
        if not _is_close(getattr(left.gaze, attr), getattr(right.gaze, attr)):
            return False
    return True


class StrictBaseModel(BaseModel):
    """Base model that rejects unexpected keys."""

    model_config = ConfigDict(extra="forbid")


class ActionUnits(StrictBaseModel):
    """Continuous Facial Action Coding System values in [0.0, 1.0]."""

    AU1_inner_brow_raiser: float = Field(..., ge=0.0, le=1.0)
    AU2_outer_brow_raiser: float = Field(..., ge=0.0, le=1.0)
    AU4_brow_lowerer: float = Field(..., ge=0.0, le=1.0)
    AU5_upper_lid_raiser: float = Field(..., ge=0.0, le=1.0)
    AU6_cheek_raiser: float = Field(..., ge=0.0, le=1.0)
    AU7_lid_tightener: float = Field(..., ge=0.0, le=1.0)
    AU9_nose_wrinkler: float = Field(..., ge=0.0, le=1.0)
    AU10_upper_lip_raiser: float = Field(..., ge=0.0, le=1.0)
    AU12_lip_corner_puller: float = Field(..., ge=0.0, le=1.0)
    AU15_lip_corner_depressor: float = Field(..., ge=0.0, le=1.0)
    AU17_chin_raiser: float = Field(..., ge=0.0, le=1.0)
    AU20_lip_stretcher: float = Field(..., ge=0.0, le=1.0)
    AU23_lip_tightener: float = Field(..., ge=0.0, le=1.0)
    AU25_lips_part: float = Field(..., ge=0.0, le=1.0)
    AU26_jaw_drop: float = Field(..., ge=0.0, le=1.0)
    AU45_blink: float = Field(..., ge=0.0, le=1.0)

    @field_validator("*", mode="before")
    @classmethod
    def reject_boolean_au_values(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("action unit values must be floats, not booleans")
        return value


class HeadPose(StrictBaseModel):
    """Subtle head pose values in image-coordinate degrees."""

    yaw: float
    pitch: float
    roll: float


class Gaze(StrictBaseModel):
    """Eye gaze values in image-coordinate degrees."""

    yaw: float
    pitch: float


class MotionState(StrictBaseModel):
    """One concrete, visible facial condition in the motion."""

    state_index: int = Field(..., ge=0)
    temporal_progress: float = Field(..., ge=0.0, le=1.0)
    expression: str
    action_units: ActionUnits
    head_pose: HeadPose
    gaze: Gaze
    visual_constraints: list[str]

    @field_validator("expression")
    @classmethod
    def expression_must_be_visible(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("expression must describe a concrete visible facial state")
        if "transition" in value.lower():
            raise ValueError("expression must not be only an abstract transition")
        return value


class GlobalConstraints(StrictBaseModel):
    """Identity, scene, and visibility constraints shared by all states."""

    same_person: bool
    same_scene: bool
    same_lighting: bool
    same_background: bool
    shoulders_and_upper_body_facing_camera: bool
    all_key_facial_features_clearly_visible: bool
    portrait_type: str
    style: str

    @model_validator(mode="after")
    def required_constraints_must_hold(self) -> "GlobalConstraints":
        required_true_fields = [
            "same_person",
            "same_scene",
            "same_lighting",
            "same_background",
            "shoulders_and_upper_body_facing_camera",
            "all_key_facial_features_clearly_visible",
        ]
        for field_name in required_true_fields:
            if getattr(self, field_name) is not True:
                raise ValueError(f"global_constraints.{field_name} must be true")
        if self.portrait_type != "head-and-shoulders realistic portrait":
            raise ValueError(
                'global_constraints.portrait_type must be "head-and-shoulders realistic portrait"'
            )
        if self.style != "natural realistic human face":
            raise ValueError('global_constraints.style must be "natural realistic human face"')
        return self


class MotionSpec(StrictBaseModel):
    """Structured facial motion specification generated from text."""

    task_type: str = "facial_motion_spec"
    input_text: str
    direction_convention: str
    global_constraints: GlobalConstraints
    states: list[MotionState] = Field(..., min_length=1, max_length=6)

    @field_validator("task_type")
    @classmethod
    def task_type_must_match(cls, value: str) -> str:
        if value != "facial_motion_spec":
            raise ValueError('task_type must be "facial_motion_spec"')
        return value

    @field_validator("input_text", "direction_convention")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("required string fields must not be empty")
        return value

    @field_validator("direction_convention")
    @classmethod
    def direction_convention_must_use_image_coordinates(cls, value: str) -> str:
        lowered = value.lower()
        if "image" not in lowered or "right side" not in lowered or "left side" not in lowered:
            raise ValueError("direction_convention must describe image-coordinate directions")
        return value

    @model_validator(mode="after")
    def states_must_be_ordered_and_complete(self) -> "MotionSpec":
        _validate_indexed_progression(self.states, "states")
        for left, right in zip(self.states, self.states[1:]):
            if _states_are_visually_identical(left, right):
                raise ValueError(
                    "adjacent MotionState entries must not be visually identical: "
                    f"state_index {left.state_index} and {right.state_index}"
                )
        return self


class EditPromptItem(StrictBaseModel):
    """One image-edit prompt corresponding to a MotionState."""

    state_index: int = Field(..., ge=0)
    temporal_progress: float = Field(..., ge=0.0, le=1.0)
    edit_prompt: str
    negative_prompt: str

    @field_validator("edit_prompt", "negative_prompt")
    @classmethod
    def prompt_text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("prompt text must not be empty")
        return value


class EditPromptSet(StrictBaseModel):
    """Compiled edit prompts for Qwen-Image-Edit-Lighting."""

    task_type: str = "motion_spec_to_edit_prompts"
    input_text: str
    prompts: list[EditPromptItem] = Field(..., min_length=1)

    @field_validator("task_type")
    @classmethod
    def task_type_must_match(cls, value: str) -> str:
        if value != "motion_spec_to_edit_prompts":
            raise ValueError('task_type must be "motion_spec_to_edit_prompts"')
        return value

    @field_validator("input_text")
    @classmethod
    def input_text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("input_text must not be empty")
        return value

    @model_validator(mode="after")
    def prompts_must_be_ordered(self) -> "EditPromptSet":
        _validate_indexed_progression(self.prompts, "prompts")
        return self
