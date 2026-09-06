from lerobot_agentic.cognitive.schemas import SpatialGroundingPlan


def test_spatial_grounding_plan_schema():
    raw_json = """{
        "sub_goal": "reach_red_cube",
        "target_object": "red_cube",
        "target_box_2d": [420, 480, 520, 580],
        "destination_box_2d": [400, 200, 480, 280],
        "confidence_score": 0.95,
        "should_halt": false,
        "reasoning": "Cube is within direct kinematic reach."
    }"""
    plan = SpatialGroundingPlan.model_validate_json(raw_json)
    assert plan.sub_goal == "reach_red_cube"
    assert len(plan.target_box_2d) == 4
    assert plan.confidence_score == 0.95
    assert not plan.should_halt
    assert not plan.requires_replanning
    assert plan.task_progress == "in_progress"

def test_spatial_grounding_plan_recovery_trigger():
    raw_json = """{
        "sub_goal": "recover",
        "target_object": "red_cube",
        "target_box_2d": [600, 300, 700, 400],
        "task_progress": "failure_detected",
        "requires_replanning": true,
        "should_halt": false
    }"""
    plan = SpatialGroundingPlan.model_validate_json(raw_json)
    assert plan.sub_goal == "recover"
    assert plan.requires_replanning
    assert plan.task_progress == "failure_detected"
