import mujoco
import numpy as np


class ClassicalIKController:
    """
    Jacobian Damped Least Squares (DLS) Inverse Kinematics controller for 6-DoF arm
    with downward-constrained gripper orientation and 7-stage Pick-and-Place state machine.
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, damping: float = 0.05):
        self.model = model
        self.data = data
        self.damping = damping
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.arm_qpos_indices = [self.model.jnt_qposadr[self.model.joint(j).id] for j in self.arm_joint_names]
        self.gripper_qpos_idx = self.model.jnt_qposadr[self.model.joint("finger_joint1").id]

        # 7-stage Pick-and-Place state machine tracking
        self.stage: str = "PREGRASP"
        self.stage_timer: int = 0

    def reset_state_machine(self):
        self.stage = "PREGRASP"
        self.stage_timer = 0

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Applies actuator-specific command limits (arm: [-pi, pi], gripper: [-0.025, 0.025])."""
        clipped = action.copy()
        clipped[:6] = np.clip(clipped[:6], -3.14159, 3.14159)
        if len(clipped) > 6:
            clipped[6] = np.clip(clipped[6], -0.025, 0.025)
        return clipped

    def solve_ik_step(
        self,
        target_pos_world: np.ndarray,
        target_rot_world: np.ndarray | None = None,
        gripper_cmd: float = 0.02,
        orientation_weight: float = 0.15,
    ) -> np.ndarray:
        """
        Computes 7-element actuator target position vector [q1..q6, q_grip].
        Implements 6D Pose IK:
        - When target_rot_world is provided: full stacked 6D DLS IK (e = [e_pos; w*e_rot], J = [J_pos; w*J_rot]).
        - When target_rot_world is None: positional DLS IK with secondary nullspace posture projection
          for downward gripper alignment without singular lockup.
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
            # Full 6D Pose IK with explicit SO(3) rotation target
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
            # Positional IK with downward-alignment nullspace projection
            lambda_sq = (self.damping ** 2) * np.eye(3)
            j_pinv = j_pos_arm.T @ np.linalg.inv(j_pos_arm @ j_pos_arm.T + lambda_sq)
            dq_pos = j_pinv @ pos_error

            # Secondary nullspace task: align wrist joint 5 downward and pan joint 1 toward target
            pan = np.arctan2(target_pos_world[1], target_pos_world[0])
            q_posture = np.array([pan, 0.6, 0.6, 0.0, 0.8, 0.0])
            current_q = np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices])
            n_proj = np.eye(6) - j_pinv @ j_pos_arm
            dq = dq_pos + n_proj @ (0.4 * (q_posture - current_q))

        current_q = np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices])
        target_arm_q = current_q + np.clip(dq, -0.08, 0.08)
        raw_action = np.concatenate([target_arm_q, [gripper_cmd]])
        return self.clip_action(raw_action)

    def step_pick_and_place(
        self,
        cube_pos: np.ndarray,
        receptacle_pos: np.ndarray | None = None
    ) -> tuple[np.ndarray, str, bool]:
        """
        Executes one 50 Hz control step of the 7-stage finite-state machine:
        PREGRASP -> APPROACH -> GRASP -> LIFT -> TRANSPORT -> PLACE -> RETREAT
        Returns (action_7d, active_stage_name, is_task_completed).
        """
        if receptacle_pos is None:
            receptacle_pos = np.array([0.32, -0.15, 0.43])
        self.stage_timer += 1
        current_ee_pos = self.data.site_xpos[self.ee_site_id] if self.ee_site_id != -1 else self.data.xpos[self.model.body("gripper_base").id]
        is_completed = False

        if self.stage == "PREGRASP":
            target = cube_pos + np.array([0.0, 0.0, 0.08])
            action = self.solve_ik_step(target, gripper_cmd=0.02)
            if np.linalg.norm(current_ee_pos - target) < 0.02 or self.stage_timer > 40:
                self.stage = "APPROACH"
                self.stage_timer = 0

        elif self.stage == "APPROACH":
            target = cube_pos + np.array([0.0, 0.0, 0.01])
            action = self.solve_ik_step(target, gripper_cmd=0.02)
            if np.linalg.norm(current_ee_pos - target) < 0.015 or self.stage_timer > 30:
                self.stage = "GRASP"
                self.stage_timer = 0

        elif self.stage == "GRASP":
            target = cube_pos + np.array([0.0, 0.0, 0.01])
            action = self.solve_ik_step(target, gripper_cmd=-0.02)
            if self.stage_timer > 25:
                self.stage = "LIFT"
                self.stage_timer = 0

        elif self.stage == "LIFT":
            target = cube_pos + np.array([0.0, 0.0, 0.12])
            action = self.solve_ik_step(target, gripper_cmd=-0.02)
            if current_ee_pos[2] > cube_pos[2] + 0.06 or self.stage_timer > 35:
                self.stage = "TRANSPORT"
                self.stage_timer = 0

        elif self.stage == "TRANSPORT":
            target = receptacle_pos + np.array([0.0, 0.0, 0.08])
            action = self.solve_ik_step(target, gripper_cmd=-0.02)
            if np.linalg.norm(current_ee_pos[:2] - target[:2]) < 0.03 or self.stage_timer > 50:
                self.stage = "PLACE"
                self.stage_timer = 0

        elif self.stage == "PLACE":
            target = receptacle_pos + np.array([0.0, 0.0, 0.02])
            action = self.solve_ik_step(target, gripper_cmd=0.02)  # Open gripper
            if self.stage_timer > 25:
                self.stage = "RETREAT"
                self.stage_timer = 0

        elif self.stage == "RETREAT":
            target = np.array([0.25, 0.0, 0.55])
            action = self.solve_ik_step(target, gripper_cmd=0.02)
            is_completed = True

        else:
            action = np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices] + [0.02])

        return action, self.stage, is_completed
