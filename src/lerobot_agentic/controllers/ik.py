import mujoco
import numpy as np


class ClassicalIKController:
    """
    Jacobian Damped Least Squares (DLS) Inverse Kinematics controller for 6-DoF arm.
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, damping: float = 0.05):
        self.model = model
        self.data = data
        self.damping = damping
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.arm_qpos_indices = [self.model.jnt_qposadr[self.model.joint(j).id] for j in self.arm_joint_names]
        self.gripper_qpos_idx = self.model.jnt_qposadr[self.model.joint("finger_joint1").id]

    def solve_ik_step(self, target_pos_world: np.ndarray, gripper_cmd: float = 0.02) -> np.ndarray:
        """Computes 7-element actuator target position vector [q1..q6, q_grip]."""
        current_ee_pos = self.data.site_xpos[self.ee_site_id] if self.ee_site_id != -1 else self.data.xpos[self.model.body("gripper_base").id]
        error = target_pos_world - current_ee_pos

        jac_pos = np.zeros((3, self.model.nv))
        if self.ee_site_id != -1:
            mujoco.mj_jacSite(self.model, self.data, jac_pos, None, self.ee_site_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jac_pos, None, self.model.body("gripper_base").id)

        # Slice 6 robot arm velocity DoFs (safe addressing)
        j_arm = jac_pos[:, :6]
        lambda_sq = (self.damping ** 2) * np.eye(3)
        inv_term = np.linalg.inv(j_arm @ j_arm.T + lambda_sq)
        dq = j_arm.T @ inv_term @ error

        current_q = np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices])
        target_arm_q = current_q + np.clip(dq, -0.08, 0.08)
        return np.concatenate([target_arm_q, [gripper_cmd]])
