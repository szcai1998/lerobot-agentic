from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SpatialGroundingPlan(BaseModel):
    """Structured spatial grounding and high-level cognitive plan emitted by Gemini Robotics ER."""
    sub_goal: Literal["reach", "grasp", "lift", "transport", "recover", "reach_cube", "grasp_cube", "lift_cube", "transport_to_zone"] = Field(
        description="Concise active sub-task primitive"
    )
    target_object: str = Field(description="Target object name, e.g., 'red_cube'")
    target_box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] normalized bounding box coordinates in [0, 1000]")
    destination_box_2d: list[int] | None = Field(default=None, description="[ymin, xmin, ymax, xmax] target drop receptacle box in [0, 1000]")
    task_progress: Literal["in_progress", "completed", "failure_detected"] = Field(
        default="in_progress", description="'in_progress', 'completed', or 'failure_detected'"
    )
    requires_replanning: bool = Field(default=False, description="Emergency or disturbance trigger: True if object was displaced or grasp failed")
    replan_id: int = Field(default=0, description="Monotonically increasing identifier for anomaly recovery events")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in spatial detection and affordance")
    should_halt: bool = Field(default=False, description="Emergency abort flag if collision or anomaly detected")
    reasoning: str | None = Field(default="", description="Brief chain of thought reasoning behind affordance choice")

    @field_validator("target_box_2d", "destination_box_2d")
    @classmethod
    def validate_box(cls, box: list[int] | None) -> list[int] | None:
        if box is None:
            return box
        if len(box) != 4:
            raise ValueError(f"Box must contain exactly 4 coordinates, got {len(box)}")
        ymin, xmin, ymax, xmax = box
        for c in (ymin, xmin, ymax, xmax):
            if not (0 <= c <= 1000):
                raise ValueError(f"Coordinate {c} outside [0, 1000] range")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError(f"Degenerate box dimensions: ymin={ymin}, ymax={ymax}, xmin={xmin}, xmax={xmax}")
        return box
