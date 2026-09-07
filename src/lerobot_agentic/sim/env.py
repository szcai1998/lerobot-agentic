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

        # Target cube object ID and joint addresses
        self.cube_body_id = self.model.body("target_cube").id
        self.palm_body_id = self.model.body("gripper_base").id
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        self.cube_joint_id = self.model.joint("cube_joint").id
        self.cube_qpos_adr = self.model.jnt_qposadr[self.cube_joint_id]
        self.cube_dof_adr = self.model.jnt_dofadr[self.cube_joint_id]

        self.f1_geom_id = self.model.geom("f1").id
        self.f2_geom_id = self.model.geom("f2").id
        self.cube_geom_id = self.model.geom("cube_geom").id

        self.rng = np.random.default_rng(42)
        self.reset()

    def set_cube_state(
        self,
        pos: np.ndarray | list[float],
        quat: np.ndarray | list[float] | None = None,
        linvel: np.ndarray | list[float] | None = None,
        angvel: np.ndarray | list[float] | None = None,
    ) -> None:
        """
        Safely sets the full 6-DoF freejoint kinematic and dynamic state of the manipuland,
        preventing residual velocity and orientation leakage across episode boundaries.
        """
        pos_arr = np.asarray(pos, dtype=np.float64)[:3]
        quat_arr = np.asarray(quat if quat is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)[:4]
        lin_arr = np.asarray(linvel if linvel is not None else [0.0, 0.0, 0.0], dtype=np.float64)[:3]
        ang_arr = np.asarray(angvel if angvel is not None else [0.0, 0.0, 0.0], dtype=np.float64)[:3]

        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = pos_arr
        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = quat_arr
        self.data.qvel[self.cube_dof_adr : self.cube_dof_adr + 3] = lin_arr
        self.data.qvel[self.cube_dof_adr + 3 : self.cube_dof_adr + 6] = ang_arr
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        seed: int | None = None,
        cube_pos: np.ndarray | list[float] | None = None,
        include_depth: bool = True,
    ) -> dict[str, Any]:
        """Resets physics state, sets initial arm pose, and applies deterministic scenario RNG."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Neutral arm pose: lifted, slightly bent elbow
        neutral_qpos = np.array([0.0, -0.4, 0.8, 0.0, 0.4, 0.0, 0.02], dtype=np.float64)
        for idx, val in zip(self.arm_qpos_indices, neutral_qpos):
            self.data.qpos[idx] = val

        self.data.ctrl[:len(neutral_qpos)] = neutral_qpos

        if cube_pos is not None:
            self.set_cube_state(cube_pos)
        elif seed is not None:
            # Calibrated reachable workspace (r <= 0.33m within arm kinematic limits)
            rx = float(self.rng.uniform(0.28, 0.325))
            ry = float(self.rng.uniform(-0.02, 0.06))
            self.set_cube_state(np.array([rx, ry, 0.43]))
        else:
            mujoco.mj_forward(self.model, self.data)

        return self.get_observation(include_depth=include_depth)

    def step(self, target_joint_pos: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Executes a 50Hz control step (10 x 0.002s physics substeps = 20ms).
        target_joint_pos: 7-element vector (6 joints + gripper width).
        """
        self.data.ctrl[:len(target_joint_pos)] = target_joint_pos
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self.get_observation(include_depth=True)
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
            "cube_lifted": bool(cube_pos[2] > 0.46),
        }
        return obs, reward, terminated, info

    def get_contact_diagnostics(
        self,
        grasp_active: bool = False,
        grasp_ref_relative_pos: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Computes instantaneous physical contact diagnostics:
        - normal contact force between fingers and cube (excluding table contacts)
        - maximum contact penetration depth
        - relative grasp slip while grasp is active
        """
        normal_force_sum = 0.0
        max_penetration = 0.0
        f_contact = np.zeros(6, dtype=np.float64)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            is_finger_cube = (
                (g1 == self.cube_geom_id and g2 in (self.f1_geom_id, self.f2_geom_id)) or
                (g2 == self.cube_geom_id and g1 in (self.f1_geom_id, self.f2_geom_id))
            )
            if is_finger_cube:
                mujoco.mj_contactForce(self.model, self.data, i, f_contact)
                # f_contact[0] is the contact normal force
                normal_force_sum += float(abs(f_contact[0]))
                if c.dist < 0:
                    max_penetration = max(max_penetration, float(-c.dist))

        slip = 0.0
        if grasp_active and grasp_ref_relative_pos is not None:
            ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3) if self.ee_site_id != -1 else np.eye(3)
            curr_rel = ee_mat.T @ (self.get_cube_position() - self.get_ee_position())
            slip = float(np.linalg.norm(curr_rel - grasp_ref_relative_pos))

        return {
            "finger_contact_force_normal": normal_force_sum,
            "penetration_depth": max_penetration,
            "relative_slip": slip,
        }

    def check_settled_task_success(
        self,
        receptacle_pos: np.ndarray | None = None,
        dist_thresh: float = 0.03,
        vel_thresh: float = 0.02,
    ) -> bool:
        """
        Evaluates physical placement success after action effect:
        Cube xy within dist_thresh of receptacle, low kinetic velocity, and resting on table.
        """
        if receptacle_pos is None:
            rec_pos = np.array([0.32, -0.15, 0.43], dtype=np.float32)
        else:
            rec_pos = np.asarray(receptacle_pos, dtype=np.float32)
        cube_pos = self.get_cube_position()
        cube_vel = self.data.qvel[self.cube_dof_adr : self.cube_dof_adr + 3]
        dist = float(np.linalg.norm(cube_pos[:2] - rec_pos[:2]))
        speed = float(np.linalg.norm(cube_vel))
        is_at_height = bool(0.41 <= cube_pos[2] <= 0.45)
        return bool(dist < dist_thresh and speed < vel_thresh and is_at_height)

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
            "ee_pos": self.get_ee_position(),
        }
        if include_depth:
            obs["depth"] = self.render_camera("overhead_cam", depth=True)
        return obs

