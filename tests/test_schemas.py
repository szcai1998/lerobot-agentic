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
        "decision_note": "Cube is within direct kinematic reach."
    }"""
    plan = SpatialGroundingPlan.model_validate_json(raw_json)
    assert plan.sub_goal == "reach"
    assert len(plan.target_box_2d) == 4
    assert plan.confidence_score == 0.95
    assert not plan.should_halt
    assert not plan.requires_replanning
    assert plan.task_progress == "in_progress"
    assert plan.replan_id == 0
    assert plan.decision_note == "Cube is within direct kinematic reach."


def test_spatial_grounding_plan_recovery_trigger():
    raw_json = """{
        "sub_goal": "recover",
        "target_object": "red_cube",
        "target_box_2d": [600, 300, 700, 400],
        "task_progress": "failure_detected",
        "requires_replanning": true,
        "replan_id": 1,
        "should_halt": false,
        "decision_note": "Object displaced outside grasp corridor"
    }"""
    plan = SpatialGroundingPlan.model_validate_json(raw_json)
    assert plan.sub_goal == "recover"
    assert plan.requires_replanning
    assert plan.replan_id == 1
    assert plan.task_progress == "failure_detected"
    assert plan.decision_note == "Object displaced outside grasp corridor"


def test_spatial_grounding_plan_literal_validation():
    # Canonical 7 primitives must all be accepted
    for sg in ["reach", "grasp", "lift", "transport", "place", "retreat", "recover"]:
        p = SpatialGroundingPlan(
            sub_goal=sg,
            target_object="red_cube",
            target_box_2d=[400, 400, 600, 600]
        )
        assert p.sub_goal == sg

    # Deprecated aliases (reach_cube, grasp_cube) must now be rejected
    for deprecated in ["reach_cube", "grasp_cube", "lift_cube", "transport_to_zone", "rotate_the_moon"]:
        with pytest.raises(ValidationError):
            SpatialGroundingPlan(
                sub_goal=deprecated,
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
