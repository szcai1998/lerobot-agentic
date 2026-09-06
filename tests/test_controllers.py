import numpy as np
import pytest

from lerobot_agentic.controllers.camera import CameraGeometry, InvalidDepthError
from lerobot_agentic.controllers.ik import ClassicalIKController
from lerobot_agentic.sim.env import MuJoCoRobotEnv


def test_camera_geometry_intrinsics_and_unprojection():
    env = MuJoCoRobotEnv()
    cam = CameraGeometry(env.model, "overhead_cam", width=640, height=480)

    # Intrinsics K matrix checks
    assert cam.K.shape == (3, 3)
    assert cam.cx == 320.0
    assert cam.cy == 240.0
    assert cam.fx > 0 and cam.fy > 0
    assert np.isclose(cam.fx, cam.fy)

    # Valid depth unprojection
    world_pos = cam.unproject_pixel_to_world(320.0, 240.0, 1.0, env.data)
    assert len(world_pos) == 3
    assert not np.any(np.isnan(world_pos))

    # INVALID_DEPTH error triggers
    with pytest.raises(InvalidDepthError):
        cam.unproject_pixel_to_world(320.0, 240.0, 0.01, env.data, min_depth=0.05)

    with pytest.raises(InvalidDepthError):
        cam.unproject_pixel_to_world(320.0, 240.0, 3.5, env.data, max_depth=2.5)

    with pytest.raises(InvalidDepthError):
        cam.unproject_pixel_to_world(320.0, 240.0, float("nan"), env.data)


def test_classical_ik_controller():
    env = MuJoCoRobotEnv()
    ik = ClassicalIKController(env.model, env.data, damping=0.05)

    # ee_site should be found
    assert ik.ee_site_id != -1
    assert len(ik.arm_qpos_indices) == 6

    # Target position near the cube
    target_pos = np.array([0.32, 0.05, 0.45], dtype=np.float64)
    cmd = ik.solve_ik_step(target_pos, gripper_cmd=0.02)
    assert len(cmd) == 7
    assert not np.any(np.isnan(cmd))
    assert np.isclose(cmd[-1], 0.02)


def test_actuator_limits_clipping():
    env = MuJoCoRobotEnv()
    ik = ClassicalIKController(env.model, env.data)

    # Extreme action should be clipped per-actuator
    extreme_action = np.array([10.0, -10.0, 5.0, -5.0, 10.0, -10.0, 1.0])
    clipped = ik.clip_action(extreme_action)

    assert np.all(clipped[:6] <= 3.14159)
    assert np.all(clipped[:6] >= -3.14159)
    assert -0.025 <= clipped[6] <= 0.025


def test_pick_and_place_fsm():
    env = MuJoCoRobotEnv()
    ik = ClassicalIKController(env.model, env.data)

    cube_pos = np.array([0.32, 0.05, 0.43])
    receptacle_pos = np.array([0.32, -0.15, 0.43])

    # Starts in PREGRASP
    assert ik.stage == "PREGRASP"
    cmd, stage, completed = ik.step_pick_and_place(cube_pos, receptacle_pos)
    assert stage == "PREGRASP"
    assert not completed
    assert len(cmd) == 7

    # Advance until stage transitions
    for _ in range(50):
        cmd, stage, _ = ik.step_pick_and_place(cube_pos, receptacle_pos)

    # Should have transitioned from PREGRASP to APPROACH
    assert stage in ["APPROACH", "GRASP"]


def test_6d_pose_ik():
    """Verifies that 6D Pose IK executes with explicit orientation matrix."""
    env = MuJoCoRobotEnv()
    ik = ClassicalIKController(env.model, env.data, damping=0.05)

    target_pos = np.array([0.32, 0.05, 0.45], dtype=np.float64)
    target_rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float64)

    cmd = ik.solve_ik_step(target_pos, target_rot_world=target_rot, gripper_cmd=0.015)
    assert len(cmd) == 7
    assert not np.any(np.isnan(cmd))
    assert np.isclose(cmd[-1], 0.015)
    assert np.all(cmd[:6] <= 3.14159)
    assert np.all(cmd[:6] >= -3.14159)
