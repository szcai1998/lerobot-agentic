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
