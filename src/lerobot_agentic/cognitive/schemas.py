
from pydantic import BaseModel, Field


class SpatialGroundingPlan(BaseModel):
    """Structured spatial grounding and high-level cognitive plan emitted by Gemini Robotics ER."""
    sub_goal: str = Field(description="Concise active sub-task (e.g., 'reach_cube', 'grasp_cube', 'lift_cube', 'transport_to_zone')")
    target_object: str = Field(description="Target object name, e.g., 'red_cube'")
    target_box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] normalized bounding box coordinates in [0, 1000]")
    destination_box_2d: list[int] | None = Field(default=None, description="[ymin, xmin, ymax, xmax] target drop receptacle box in [0, 1000]")
    task_progress: str = Field(default="in_progress", description="'in_progress', 'completed', or 'failure_detected'")
    requires_replanning: bool = Field(default=False, description="Emergency or disturbance trigger: True if object was displaced or grasp failed")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in spatial detection and affordance")
    should_halt: bool = Field(default=False, description="Emergency abort flag if collision or anomaly detected")
    reasoning: str | None = Field(default="", description="Brief chain of thought reasoning behind affordance choice")
