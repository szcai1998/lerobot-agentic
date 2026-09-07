import json
from pathlib import Path
from typing import Any

import numpy as np


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Computes Wilson 95% Score Confidence Interval for binary success proportions.
    Provides asymmetric, non-truncated interval [0, 1] especially robust for small N.
    """
    if total <= 0:
        return 0.0, 0.0

    p_hat = float(successes) / float(total)
    # Critical value for two-sided confidence
    z = 1.95996  # for 95% CI
    if confidence != 0.95:
        from scipy.stats import norm
        z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)

    denom = 1.0 + (z ** 2) / total
    center = (p_hat + (z ** 2) / (2.0 * total)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / total) + ((z ** 2) / (4.0 * (total ** 2))))

    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return lower, upper


def bootstrap_ci(
    data: np.ndarray,
    n_resamples: int = 10000,
    ci: float = 0.95,
    random_state: int | None = 42
) -> tuple[float, float]:
    """
    Computes non-parametric bootstrap confidence interval for continuous metric distributions.
    """
    arr = np.asarray(data).flatten()
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(random_state)
    boot_indices = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    boot_means = np.mean(arr[boot_indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return lower, upper


def compute_trajectory_jerk(q_traj: np.ndarray, dt: float = 0.02) -> float:
    """
    Computes mean joint trajectory jerk (third time-derivative of position) in rad/s^3.
    Proxy for motion smoothness and aggressive actuator-command variation.
    q_traj: (T, dof) array of joint positions.
    """
    q = np.asarray(q_traj)
    if len(q) < 4:
        return 0.0

    # 3rd central-difference approximation
    jerk_vectors = (q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]) / (dt ** 3)
    jerk_norms = np.linalg.norm(jerk_vectors, axis=1)
    return float(np.mean(jerk_norms))


def compute_joint_and_gripper_jerk(q_traj: np.ndarray, dt: float = 0.02) -> dict[str, float]:
    """
    Computes RMS trajectory jerk with strict unit separation:
    - Arm joints 1..6 (revolute): rad/s^3
    - Gripper joint 7 (prismatic): m/s^3
    q_traj: (T, 7) array where columns 0..5 are arm revolute angles (rad)
            and column 6 is prismatic gripper position (m).
    """
    q = np.asarray(q_traj)
    if len(q) < 4:
        return {
            "arm_joint_jerk_rms_rad_s3": 0.0,
            "gripper_jerk_rms_m_s3": 0.0,
        }

    # 3rd central-difference approximation: (q[t+3] - 3q[t+2] + 3q[t+1] - q[t]) / dt^3
    jerk = (q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]) / (dt ** 3)

    # 1. Arm revolute joints (indices 0..5): Frobenius norm across joints, RMS over time
    arm_jerk_slice = jerk[:, :6]
    arm_jerk_norms = np.linalg.norm(arm_jerk_slice, axis=1)
    arm_rms = float(np.sqrt(np.mean(arm_jerk_norms ** 2)))

    # 2. Gripper prismatic finger (index 6): RMS over time
    if q.shape[1] > 6:
        gripper_jerk = jerk[:, 6]
        gripper_rms = float(np.sqrt(np.mean(gripper_jerk ** 2)))
    else:
        gripper_rms = 0.0

    return {
        "arm_joint_jerk_rms_rad_s3": round(arm_rms, 2),
        "gripper_jerk_rms_m_s3": round(gripper_rms, 4),
    }


class ScenarioManifestLogger:
    """
    Logs complete scenario provenance records to scenario_manifest.jsonl for reproducibility.
    """
    def __init__(self, output_path: str = "outputs/benchmarks/scenario_manifest.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log_scenario(
        self,
        seed: int,
        system_id: str,
        cube_initial_pose: list[float],
        target_zone_pose: list[float],
        distractor_poses: list[list[float]],
        texture_seed: int,
        lighting_vector: list[float],
        disturbance_time_step: int | None = None,
        disturbance_displacement: list[float] | None = None,
        model_name: str = "gemini-robotics-er-2-preview",
        checkpoint_sha256: str | None = None,
        git_commit_hash: str | None = None,
        extra: dict[str, Any] | None = None
    ):
        record = {
            "seed": seed,
            "system_id": system_id,
            "cube_initial_pose": cube_initial_pose,
            "target_zone_pose": target_zone_pose,
            "distractor_poses": distractor_poses,
            "texture_seed": texture_seed,
            "lighting_vector": lighting_vector,
            "disturbance_time_step": disturbance_time_step,
            "disturbance_displacement": disturbance_displacement,
            "model_name": model_name,
            "checkpoint_sha256": checkpoint_sha256,
            "git_commit_hash": git_commit_hash,
            **(extra or {})
        }
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
