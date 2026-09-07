from dataclasses import dataclass
from typing import Any

import cv2
import mujoco
import numpy as np

from lerobot_agentic.controllers.camera import CameraGeometry

CANONICAL_SUBGOALS = ["reach", "grasp", "lift", "transport", "place", "retreat", "recover"]
SUBGOAL_TO_IDX = {sg: i for i, sg in enumerate(CANONICAL_SUBGOALS)}

FSM_TO_CANONICAL = {
    "PREGRASP": "reach",
    "APPROACH": "reach",
    "GRASP": "grasp",
    "LIFT": "lift",
    "TRANSPORT": "transport",
    "PLACE": "place",
    "RETREAT": "retreat",
}


@dataclass(frozen=True)
class ExpertStep:
    action: np.ndarray             # (7,) target joint positions [q1..q6, q_grip]
    fsm_stage: str                 # PREGRASP, APPROACH, GRASP, LIFT, TRANSPORT, PLACE, RETREAT
    canonical_subgoal: str         # reach, grasp, lift, transport, place, retreat, recover
    goal_snapshot: np.ndarray      # (13,) [target_obs_3d, dest_3d, one_hot_7d]
    controller_terminal: bool      # FSM completed RETREAT
    diagnostics: dict[str, float]  # Instantaneous perception & contact metrics


class ClassicalIKController:
    """
    Jacobian Damped Least Squares (DLS) Inverse Kinematics controller for 6-DoF arm
    with downward-constrained gripper orientation, 7-stage Pick-and-Place FSM,
    and strict separation between privileged expert actions and observable policy conditioning.
    """
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        damping: float = 0.05,
        receptacle_pos: np.ndarray | None = None,
    ):
        self.model = model
        self.data = data
        self.damping = damping
        self.receptacle_pos = np.asarray(receptacle_pos if receptacle_pos is not None else [0.295, -0.140, 0.44], dtype=np.float32)
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")

        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.arm_qpos_indices = [self.model.jnt_qposadr[self.model.joint(j).id] for j in self.arm_joint_names]
        self.gripper_qpos_idx = self.model.jnt_qposadr[self.model.joint("finger_joint1").id]

        # Camera unprojection engine for observable target estimation
        self.cam_geom = CameraGeometry(self.model, "overhead_cam", width=640, height=480)

        # 7-stage Pick-and-Place state machine tracking
        self.stage: str = "PREGRASP"
        self.stage_timer: int = 0
        self.sub_stage_timer: int = 0
        self.active_canonical_subgoal: str | None = None
        self.goal_snapshot: np.ndarray | None = None
        self.last_observed_target: np.ndarray | None = None

        # Fixed target poses captured for smooth interpolation
        self.init_cube_gt: np.ndarray | None = None
        self.lift_ee_pos: np.ndarray | None = None

        # Contact & slip physics tracking
        self.grasp_active: bool = False
        self.grasp_ref_relative_pos: np.ndarray | None = None

    def reset_state_machine(self):
        self.stage = "PREGRASP"
        self.stage_timer = 0
        self.sub_stage_timer = 0
        self.active_canonical_subgoal = None
        self.goal_snapshot = None
        self.last_observed_target = None
        self.init_cube_gt = None
        self.lift_ee_pos = None
        self.grasp_active = False
        self.grasp_ref_relative_pos = None

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Applies actuator-specific command limits (arm: [-pi, pi], gripper: [-0.025, 0.025])."""
        clipped = action.copy()
        clipped[:6] = np.clip(clipped[:6], -3.14159, 3.14159)
        if len(clipped) > 6:
            clipped[6] = np.clip(clipped[6], -0.025, 0.025)
        return clipped.astype(np.float32)

    def get_gt_target(self) -> np.ndarray:
        """Privileged access to MuJoCo ground-truth cube position (used exclusively for expert IK)."""
        return np.array(self.data.xpos[self.cube_body_id], dtype=np.float32)

    def estimate_target_from_rgbd(self, obs: dict[str, Any]) -> np.ndarray:
        """
        Non-privileged observable target estimation:
        Extracts 2D red cube centroid from overhead RGB frame, looks up metric depth,
        and unprojects to 3D Cartesian coordinates via CameraGeometry.
        """
        rgb = obs["rgb"]
        depth = obs.get("depth")
        if depth is None:
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera="overhead_cam")
            depth = renderer.render()
            renderer.close()

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255]))
        mask = mask1 | mask2

        M = cv2.moments(mask)
        if M["m00"] > 0:
            u = float(M["m10"] / M["m00"])
            v = float(M["m01"] / M["m00"])
            u_clamped = int(np.clip(round(u), 0, 639))
            v_clamped = int(np.clip(round(v), 0, 479))
            d = float(depth[v_clamped, u_clamped])
            world_pos = self.cam_geom.unproject_pixel_to_world(u, v, d, self.data)
            # Subtract cube half-height (0.022m) since overhead ray strikes top surface
            world_pos[2] -= 0.022
            self.last_observed_target = world_pos.astype(np.float32)
            return self.last_observed_target

        if self.last_observed_target is not None:
            return self.last_observed_target

        return np.array([0.32, 0.05, 0.43], dtype=np.float32)

    def solve_ik_step(
        self,
        target_pos_world: np.ndarray,
        target_rot_world: np.ndarray | None = None,
        gripper_cmd: float = 0.025,
        orientation_weight: float = 0.15,
    ) -> np.ndarray:
        """
        Computes 7-element actuator target position vector [q1..q6, q_grip].
        Uses positional DLS IK with calibrated gentle nullspace posture task.
        """
        current_ee_pos = self.data.site_xpos[self.ee_site_id] if self.ee_site_id != -1 else self.data.xpos[self.model.body("gripper_base").id]
        pos_error = target_pos_world - current_ee_pos

        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        if self.ee_site_id != -1:
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self.model.body("gripper_base").id)

        j_pos_arm = jac_pos[:, :6]
        j_rot_arm = jac_rot[:, :6]

        if target_rot_world is not None:
            current_ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3) if self.ee_site_id != -1 else self.data.xmat[self.model.body("gripper_base").id].reshape(3, 3)
            rot_error = 0.5 * (
                np.cross(current_ee_mat[:, 0], target_rot_world[:, 0]) +
                np.cross(current_ee_mat[:, 1], target_rot_world[:, 1]) +
                np.cross(current_ee_mat[:, 2], target_rot_world[:, 2])
            )
            w = orientation_weight
            j_6d = np.vstack([j_pos_arm, w * j_rot_arm])
            e_6d = np.concatenate([pos_error, w * rot_error])
            lambda_sq = (self.damping ** 2) * np.eye(6)
            dq = j_6d.T @ np.linalg.inv(j_6d @ j_6d.T + lambda_sq) @ e_6d
        else:
            lambda_sq = (self.damping ** 2) * np.eye(3)
            j_pinv = j_pos_arm.T @ np.linalg.inv(j_pos_arm @ j_pos_arm.T + lambda_sq)
            dq_pos = j_pinv @ pos_error

            pan = np.arctan2(target_pos_world[1], target_pos_world[0])
            q_posture = np.array([pan, 0.8, 0.6, 0.0, 0.8, 0.0])
            current_q = np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices])
            n_proj = np.eye(6) - j_pinv @ j_pos_arm
            null_task = n_proj @ (0.05 * (q_posture - current_q))
            dq = dq_pos + null_task

        current_q = np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices])
        target_arm_q = current_q + np.clip(dq, -0.05, 0.05)
        raw_action = np.concatenate([target_arm_q, [gripper_cmd]])
        return self.clip_action(raw_action)

    def act(self, obs: dict[str, Any]) -> ExpertStep:
        """
        Atomic 50 Hz controller execution:
        1. Inspects current state and determines FSM stage transition.
        2. Resolves canonical policy subgoal (reach, grasp, lift, transport, place, retreat).
        3. Refreshes 13-DoF goal snapshot strictly upon canonical transitions using observable RGB-D perception.
        4. Computes expert action using privileged MuJoCo GT state for the 6D IK solver.
        5. Returns structured ExpertStep dataclass.
        """
        self.stage_timer += 1
        current_ee_pos = self.data.site_xpos[self.ee_site_id] if self.ee_site_id != -1 else self.data.xpos[self.model.body("gripper_base").id]
        cube_gt = self.get_gt_target()
        receptacle_pos = np.array([0.295, -0.140, 0.52], dtype=np.float32)

        if self.init_cube_gt is None:
            self.init_cube_gt = cube_gt.copy()

        # Observable target unprojection (used strictly for policy conditioning)
        target_obs = self.estimate_target_from_rgbd(obs)

        # 1. State Machine Stage Transitions & Action Synthesis
        controller_terminal = False
        gripper_cmd = 0.025
        target_pos = self.init_cube_gt + np.array([0.0, 0.0, 0.08], dtype=np.float32)

        if self.stage == "PREGRASP":
            target_pos = self.init_cube_gt + np.array([0.0, 0.0, 0.08], dtype=np.float32)
            gripper_cmd = 0.025
            if np.linalg.norm(current_ee_pos - target_pos) < 0.02 or self.stage_timer >= 50:
                self.stage = "APPROACH"
                self.stage_timer = 0

        elif self.stage == "APPROACH":
            target_pos = self.init_cube_gt + np.array([0.0, 0.0, 0.01], dtype=np.float32)
            gripper_cmd = 0.025
            if np.linalg.norm(current_ee_pos - target_pos) < 0.015 or self.stage_timer >= 50:
                self.stage = "GRASP"
                self.stage_timer = 0

        elif self.stage == "GRASP":
            target_pos = self.init_cube_gt + np.array([0.0, 0.0, 0.01], dtype=np.float32)
            gripper_cmd = -0.025
            if self.stage_timer >= 30:
                self.stage = "LIFT"
                self.stage_timer = 0
                self.lift_ee_pos = current_ee_pos.copy()
                self.grasp_active = True
                ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3) if self.ee_site_id != -1 else np.eye(3)
                self.grasp_ref_relative_pos = ee_mat.T @ (cube_gt - current_ee_pos)

        elif self.stage == "LIFT":
            target_pos = self.init_cube_gt + np.array([0.0, 0.0, 0.12], dtype=np.float32)
            gripper_cmd = -0.025
            if self.stage_timer >= 45:
                self.stage = "TRANSPORT"
                self.stage_timer = 0
                self.lift_ee_pos = current_ee_pos.copy()

        elif self.stage == "TRANSPORT":
            # Smooth quintic interpolation between lift end-effector pose and receptacle zone
            gripper_cmd = -0.025
            alpha = min(1.0, self.stage_timer / 59.0)
            tau = 10 * (alpha**3) - 15 * (alpha**4) + 6 * (alpha**5)
            start_pos = self.lift_ee_pos if self.lift_ee_pos is not None else (self.init_cube_gt + np.array([0.0, 0.0, 0.12], dtype=np.float32))
            target_pos = (1.0 - tau) * start_pos + tau * receptacle_pos
            if self.stage_timer >= 60:
                self.stage = "PLACE"
                self.stage_timer = 0
                self.sub_stage_timer = 0
                self.grasp_active = False

        elif self.stage == "PLACE":
            place_target = np.array([0.295, -0.140, 0.44], dtype=np.float32)
            if self.stage_timer <= 30:
                # Part 1: lower cube onto receptacle surface
                alpha = min(1.0, self.stage_timer / 29.0)
                target_pos = (1.0 - alpha) * receptacle_pos + alpha * place_target
                gripper_cmd = -0.025
            else:
                # Part 2: open gripper fingers to deposit cube
                target_pos = place_target
                gripper_cmd = 0.025
                self.grasp_active = False

            if self.stage_timer >= 60:
                self.stage = "RETREAT"
                self.stage_timer = 0

        elif self.stage == "RETREAT":
            # Smooth vertical lift for 35 steps away from table before horizontal return
            gripper_cmd = 0.025
            self.grasp_active = False
            if self.stage_timer <= 35:
                alpha = min(1.0, self.stage_timer / 34.0)
                target_pos = (1.0 - alpha) * np.array([0.295, -0.140, 0.44], dtype=np.float32) + alpha * np.array([0.295, -0.140, 0.55], dtype=np.float32)
            else:
                target_pos = np.array([0.25, 0.0, 0.55], dtype=np.float32)

            if self.stage_timer >= 60:
                controller_terminal = True

        # 2. Canonical Subgoal Mapping & Snapshot Refresh
        canonical_subgoal = FSM_TO_CANONICAL[self.stage]
        if self.goal_snapshot is None or canonical_subgoal != self.active_canonical_subgoal:
            self.active_canonical_subgoal = canonical_subgoal
            vec = np.zeros(13, dtype=np.float32)
            vec[:3] = target_obs[:3]
            vec[3:6] = self.receptacle_pos[:3]
            vec[6 + SUBGOAL_TO_IDX[canonical_subgoal]] = 1.0
            self.goal_snapshot = vec

        # 3. Compute 6D IK action
        action = self.solve_ik_step(target_pos, gripper_cmd=gripper_cmd)

        # 4. Perception & State Diagnostics
        perception_err = float(np.linalg.norm(target_obs - cube_gt))
        diagnostics = {
            "perception_error_m": perception_err,
            "stage_timer": float(self.stage_timer),
            "grasp_active": float(self.grasp_active),
        }

        return ExpertStep(
            action=action,
            fsm_stage=self.stage,
            canonical_subgoal=canonical_subgoal,
            goal_snapshot=self.goal_snapshot.copy(),
            controller_terminal=controller_terminal,
            diagnostics=diagnostics,
        )

    def step_pick_and_place(
        self,
        cube_pos: np.ndarray,
        receptacle_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str, bool]:
        """Legacy interface for backwards-compatibility."""
        if receptacle_pos is not None:
            self.receptacle_pos = np.asarray(receptacle_pos, dtype=np.float32)
        obs = {"rgb": np.zeros((480, 640, 3), dtype=np.uint8), "depth": np.zeros((480, 640), dtype=np.float32)}
        step = self.act(obs)
        return step.action, step.fsm_stage, step.controller_terminal
