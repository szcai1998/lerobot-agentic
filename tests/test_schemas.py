import pytest
from pydantic import ValidationError

from lerobot_agentic.cognitive.schemas import SpatialGroundingPlan


def test_spatial_grounding_plan_schema():
    raw_json = """{
        "sub_goal": "reach",
        "target_object": "red_cube",
        "target_box_2d": [420, 480, 520, 580],
        "destination_box_2d": [400, 200, 480, 280],
        "confidence_score": 0.95,
        "should_halt": false,
        "reasoning": "Cube is within direct kinematic reach."
    }"""
    plan = SpatialGroundingPlan.model_validate_json(raw_json)
    assert plan.sub_goal == "reach"
    assert len(plan.target_box_2d) == 4
    assert plan.confidence_score == 0.95
    assert not plan.should_halt
    assert not plan.requires_replanning
    assert plan.task_progress == "in_progress"
    assert plan.replan_id == 0


def test_spatial_grounding_plan_recovery_trigger():
    raw_json = """{
        "sub_goal": "recover",
        "target_object": "red_cube",
        "target_box_2d": [600, 300, 700, 400],
        "task_progress": "failure_detected",
        "requires_replanning": true,
        "replan_id": 1,
        "should_halt": false
    }"""
    plan = SpatialGroundingPlan.model_validate_json(raw_json)
    assert plan.sub_goal == "recover"
    assert plan.requires_replanning
    assert plan.replan_id == 1
    assert plan.task_progress == "failure_detected"


def test_spatial_grounding_plan_literal_validation():
    # Invalid sub_goal must be rejected at schema validation
    with pytest.raises(ValidationError):
        SpatialGroundingPlan(
            sub_goal="rotate_the_moon",
            target_object="red_cube",
            target_box_2d=[400, 400, 600, 600]
        )

    # Invalid task_progress must be rejected
    with pytest.raises(ValidationError):
        SpatialGroundingPlan(
            sub_goal="reach",
            target_object="red_cube",
            target_box_2d=[400, 400, 600, 600],
            task_progress="unknown_state"
        )


def test_spatial_grounding_plan_box_validation():
    # Degenerate box coordinates (ymin >= ymax) must be rejected
    with pytest.raises(ValidationError):
        SpatialGroundingPlan(
            sub_goal="reach",
            target_object="red_cube",
            target_box_2d=[600, 400, 400, 600]
        )

    # Box coordinates outside [0, 1000] must be rejected
    with pytest.raises(ValidationError):
        SpatialGroundingPlan(
            sub_goal="reach",
            target_object="red_cube",
            target_box_2d=[-10, 400, 400, 600]
        )
