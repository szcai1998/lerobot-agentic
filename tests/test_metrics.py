import json

import numpy as np

from lerobot_agentic.cognitive.plan_state import AtomicPlanState, LatestFrameBuffer
from lerobot_agentic.cognitive.schemas import SpatialGroundingPlan
from lerobot_agentic.utils.metrics import (
    ScenarioManifestLogger,
    bootstrap_ci,
    compute_trajectory_jerk,
    wilson_score_interval,
)


def test_wilson_score_interval():
    low, high = wilson_score_interval(18, 20, confidence=0.95)
    assert 0.0 <= low <= high <= 1.0
    assert low > 0.65  # 18/20 = 90%
    assert high < 0.99

    # Boundary cases
    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    low_0, high_0 = wilson_score_interval(0, 20)
    assert low_0 == 0.0
    assert high_0 > 0.0  # Wilson interval gives positive upper bound even for 0 successes


def test_bootstrap_ci():
    data = np.array([1.0, 1.2, 1.1, 0.9, 1.05, 1.15, 0.95])
    low, high = bootstrap_ci(data, n_resamples=1000, ci=0.95)
    mean = np.mean(data)
    assert low <= mean <= high


def test_compute_trajectory_jerk():
    # Constant position -> zero jerk
    constant_traj = np.ones((20, 7))
    jerk = compute_trajectory_jerk(constant_traj, dt=0.02)
    assert np.isclose(jerk, 0.0)

    # Dynamic trajectory -> positive jerk
    t = np.linspace(0, 1.0, 20)[:, None]
    dyn_traj = np.sin(2 * np.pi * t) * np.ones((20, 7))
    jerk_dyn = compute_trajectory_jerk(dyn_traj, dt=0.02)
    assert jerk_dyn > 0.0


def test_scenario_manifest_logger(tmp_path):
    log_file = tmp_path / "test_manifest.jsonl"
    logger = ScenarioManifestLogger(output_path=str(log_file))

    logger.log_scenario(
        seed=42,
        system_id="system_d",
        cube_initial_pose=[0.32, 0.05, 0.43],
        target_zone_pose=[0.32, -0.15, 0.40],
        distractor_poses=[],
        texture_seed=101,
        lighting_vector=[0.0, 0.0, -1.0],
        disturbance_time_step=100,
        disturbance_displacement=[0.08, -0.06, 0.0],
        model_name="gemini-robotics-er-2-preview"
    )

    assert log_file.exists()
    with open(log_file) as f:
        line = f.readline()
        record = json.loads(line)
        assert record["seed"] == 42
        assert record["system_id"] == "system_d"
        assert record["disturbance_displacement"] == [0.08, -0.06, 0.0]


def test_atomic_plan_state_concurrency():
    state = AtomicPlanState()
    assert state.version == 0
    assert state.get_snapshot()[0] is None

    plan = SpatialGroundingPlan(
        sub_goal="reach_cube",
        target_object="red_cube",
        target_box_2d=[400, 400, 600, 600],
        task_progress="in_progress",
        requires_replanning=False
    )
    target_pos = np.array([0.3, 0.1, 0.4])
    dest_pos = np.array([0.3, -0.2, 0.4])

    state.update(plan, target_pos, dest_pos, subgoal_id=0)
    snap_plan, snap_t, snap_d, snap_sub, snap_v = state.get_snapshot()

    assert snap_v == 1
    assert snap_plan.sub_goal == "reach_cube"
    assert np.allclose(snap_t, target_pos)
    assert np.allclose(snap_d, dest_pos)
    assert snap_sub == 0


def test_latest_frame_buffer():
    buf = LatestFrameBuffer()
    assert buf.get_latest()[0] is None

    rgb_mock = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_mock = np.ones((480, 640), dtype=np.float32) * 0.75
    buf.push(rgb_mock, depth_mock, step_idx=10)

    rgb_out, depth_out, wrist_out, step_idx = buf.get_latest()
    assert rgb_out is not None
    assert depth_out is not None
    assert wrist_out is None
    assert step_idx == 10
    assert np.allclose(depth_out, 0.75)


def test_edge_triggered_replan():
    state = AtomicPlanState()
    # No replan initially
    consumed_id = 0
    needs_replan, consumed_id = state.check_and_consume_replan(consumed_id)
    assert not needs_replan
    assert consumed_id == 0

    # Disturbance occurs -> replan triggered with id 1
    plan = SpatialGroundingPlan(
        sub_goal="recover",
        target_object="red_cube",
        target_box_2d=[400, 400, 600, 600],
        task_progress="failure_detected",
        requires_replanning=True,
        replan_id=1
    )
    state.update(plan)

    # First check: rising edge detected
    needs_replan, consumed_id = state.check_and_consume_replan(consumed_id)
    assert needs_replan
    assert consumed_id == 1

    # Second check (same plan state still has requires_replanning=True): already consumed!
    needs_replan, consumed_id = state.check_and_consume_replan(consumed_id)
    assert not needs_replan
    assert consumed_id == 1

