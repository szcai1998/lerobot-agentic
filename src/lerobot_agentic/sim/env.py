import os
import pathlib
from typing import Any

import numpy as np

# Force EGL rendering on headless Linux servers/desktops if display is absent or requested
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco

DEFAULT_MODEL_PATH = pathlib.Path(__file__).parent / "models" / "embodied_arm.xml"


class MuJoCoRobotEnv:
    """
    Deterministic Physics Simulation Environment using DeepMind MuJoCo.
    Encapsulates 6-DOF manipulation arm + parallel gripper + overhead/wrist camera sensors.
    """
    def __init__(self, xml_path: str | None = None):
        self.xml_path = str(xml_path or DEFAULT_MODEL_PATH)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Cache joint addresses to guarantee safe indexing regardless of freejoints
        self.arm_joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6",
            "finger_joint1"
        ]
        self.arm_qpos_indices = [
            self.model.jnt_qposadr[self.model.joint(name).id] 
            for name in self.arm_joint_names
        ]

        # Target cube object ID
        self.cube_body_id = self.model.body("target_cube").id
        self.palm_body_id = self.model.body("gripper_base").id
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        self.reset()

    def reset(self) -> dict[str, Any]:
        """Resets physics state and sets initial arm pose."""
        mujoco.mj_resetData(self.model, self.data)

        # Neutral arm pose: lifted, slightly bent elbow
        neutral_qpos = np.array([0.0, -0.4, 0.8, 0.0, 0.4, 0.0, 0.02], dtype=np.float64)
        for idx, val in zip(self.arm_qpos_indices, neutral_qpos):
            self.data.qpos[idx] = val

        self.data.ctrl[:len(neutral_qpos)] = neutral_qpos
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, target_joint_pos: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Executes a 50Hz control step (10 x 0.002s physics substeps = 20ms).
        target_joint_pos: 7-element vector (6 joints + gripper width).
        """
        self.data.ctrl[:len(target_joint_pos)] = target_joint_pos
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self.get_observation()
        cube_pos = self.get_cube_position()
        palm_pos = self.get_gripper_palm_position()
        dist = np.linalg.norm(cube_pos - palm_pos)

        # Reward formulation: distance reduction + cube height lift
        reward = -float(dist) + float(cube_pos[2] - 0.43) * 5.0
        terminated = False
        info = {
            "cube_position": cube_pos,
            "gripper_position": palm_pos,
            "distance_to_cube": dist,
            "cube_lifted": bool(cube_pos[2] > 0.46)
        }
        return obs, reward, terminated, info

    def render_camera(self, camera_name: str = "overhead_cam", depth: bool = False) -> np.ndarray:
        """Renders RGB or metric depth camera frame from simulation."""
        if depth:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera=camera_name)
            res = self.renderer.render()
            self.renderer.disable_depth_rendering()
            return res
        else:
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera=camera_name)
            return self.renderer.render()

    def get_proprioception(self) -> np.ndarray:
        """Returns 7-element proprioceptive joint positions."""
        return np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices], dtype=np.float32)

    def get_cube_position(self) -> np.ndarray:
        """Returns [x, y, z] Cartesian position of target cube."""
        return np.array(self.data.xpos[self.cube_body_id], dtype=np.float32)

    def get_gripper_palm_position(self) -> np.ndarray:
        """Returns [x, y, z] Cartesian position of gripper palm."""
        return np.array(self.data.xpos[self.palm_body_id], dtype=np.float32)

    def get_ee_position(self) -> np.ndarray:
        """Returns [x, y, z] Cartesian position of end-effector site."""
        if self.ee_site_id != -1:
            return np.array(self.data.site_xpos[self.ee_site_id], dtype=np.float32)
        return self.get_gripper_palm_position()

    def get_observation(self, include_depth: bool = False) -> dict[str, Any]:
        obs = {
            "rgb": self.render_camera("overhead_cam"),
            "rgb_wrist": self.render_camera("wrist_cam"),
            "proprioception": self.get_proprioception(),
            "cube_pos": self.get_cube_position(),
            "palm_pos": self.get_gripper_palm_position(),
            "ee_pos": self.get_ee_position()
        }
        if include_depth:
            obs["depth"] = self.render_camera("overhead_cam", depth=True)
        return obs
