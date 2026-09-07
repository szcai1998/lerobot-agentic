import os
import time
from typing import Any

import numpy as np
import torch

CANONICAL_SUBGOALS = ["reach", "grasp", "lift", "transport", "place", "retreat", "recover"]
SUBGOAL_TO_IDX = {sg: i for i, sg in enumerate(CANONICAL_SUBGOALS)}


def encode_goal_vector(
    target_pos: np.ndarray | list[float],
    dest_pos: np.ndarray | list[float] | None = None,
    sub_goal: str = "reach",
) -> np.ndarray:
    """
    Constructs the canonical 13-DoF goal conditioning vector:
    [p_target (3), p_dest (3), e_subgoal (7)]
    for stock LeRobot ACT observation.environment_state (FeatureType.ENV).
    """
    vec = np.zeros(13, dtype=np.float32)
    vec[:3] = np.asarray(target_pos, dtype=np.float32)[:3]
    if dest_pos is not None:
        vec[3:6] = np.asarray(dest_pos, dtype=np.float32)[:3]
    idx = SUBGOAL_TO_IDX.get(sub_goal, 0)
    vec[6 + idx] = 1.0
    return vec


class VisuomotorPolicyExecutor:
    """
    Manages inference and action chunk execution for LeRobot policies (ACT, Diffusion, SmolVLA).
    
    Primary Benchmark Execution Mode:
    - Queue / Receding Horizon (chunk_size=50, n_action_steps=10, temporal_ensemble_coeff=None)
    - Control frequency: 50 Hz, nominal inference cadence: ~5 Hz (every 10 steps).
    - Recovery: policy.reset() flushes cached action queue and triggers immediate inference.
    - Optional Ablation: 50 Hz temporal ensembling.
    """
    def __init__(
        self,
        pretrained_policy_path: str | None = None,
        chunk_size: int = 50,
        n_action_steps: int = 10,
        action_dim: int = 7,
        device: str | None = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None

        if pretrained_policy_path and os.path.exists(pretrained_policy_path):
            try:
                from lerobot.policies.act import ACTPolicy
            except ImportError:
                from lerobot.policies.act.modeling_act import ACTPolicy

            self.policy = ACTPolicy.from_pretrained(pretrained_policy_path).to(self.device)
            self.policy.eval()
            self.policy.reset()

            # Official LeRobot 0.6+ factory for restoring checkpoint pre/post-processors
            try:
                from lerobot.policies.factory import make_pre_post_processors
                self.preprocessor, self.postprocessor = make_pre_post_processors(
                    policy_cfg=self.policy.config,
                    pretrained_path=pretrained_policy_path,
                )
                print(f"[PolicyExecutor] Restored pre/post processors from {pretrained_policy_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load policy pre/post-processors from checkpoint {pretrained_policy_path}: {e}"
                ) from e

            print(f"[PolicyExecutor] Loaded Hugging Face LeRobot ACTPolicy from {pretrained_policy_path}")

    def reset(self):
        """Flushes any cached action chunk queue in the policy during replanning."""
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def environment_processor(
        self,
        rgb_top: np.ndarray,
        proprioception: np.ndarray,
        rgb_wrist: np.ndarray | None = None,
        goal_vector: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Stage 1: Raw MuJoCo observation -> Environment Processor.
        Produces UNBATCHED tensors:
        - images: (C, H, W)
        - state: (7,)
        - environment_state: (13,)
        The LeRobot preprocessor pipeline owns batching (via AddBatchDimensionProcessorStep).
        """
        batch = {
            "observation.images.top": (torch.from_numpy(rgb_top).permute(2, 0, 1).float() / 255.0).to(self.device),
            "observation.state": torch.from_numpy(proprioception).float().to(self.device),
        }
        if rgb_wrist is not None:
            batch["observation.images.wrist"] = (torch.from_numpy(rgb_wrist).permute(2, 0, 1).float() / 255.0).to(self.device)
        if goal_vector is not None:
            # Stock LeRobot ACT consumes 13-DoF environment state via FeatureType.ENV
            batch["observation.environment_state"] = torch.from_numpy(goal_vector).float().to(self.device)
        return batch

    def environment_action_adapter(self, action: Any) -> np.ndarray:
        """
        Stage 5: Environment / Action Adapter -> MuJoCo actuator command.
        Converts postprocessed tensor to numpy joint command with actuator-specific clipping:
        Arm joints 1..6 clip to [-pi, pi], gripper finger clips to [-0.025, 0.025].
        """
        if isinstance(action, torch.Tensor):
            if action.dim() > 1:
                action_np = action.squeeze(0).detach().cpu().numpy()
            else:
                action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)
        clipped = action_np.copy()
        clipped[:6] = np.clip(clipped[:6], -3.14159, 3.14159)
        if len(clipped) > 6:
            clipped[6] = np.clip(clipped[6], -0.025, 0.025)
        return clipped

    def select_action(
        self,
        rgb_top: np.ndarray,
        proprioception: np.ndarray,
        rgb_wrist: np.ndarray | None = None,
        goal_vector: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Executes single 50 Hz control step following the LeRobot 0.6+ PolicyProcessorPipeline:
        raw MuJoCo obs -> env processor -> policy preprocessor -> ACT select_action() -> postprocessor -> action adapter.
        """
        if self.policy is not None:
            # 1. Environment Processor
            batch = self.environment_processor(rgb_top, proprioception, rgb_wrist, goal_vector)

            # 2. LeRobot Policy Preprocessor (externalized normalization & batching)
            if self.preprocessor is not None:
                batch = self.preprocessor(batch)
            else:
                batch["observation.images.top"] = (batch["observation.images.top"].float() / 255.0).unsqueeze(0)
                if "observation.images.wrist" in batch:
                    batch["observation.images.wrist"] = (batch["observation.images.wrist"].float() / 255.0).unsqueeze(0)
                batch["observation.state"] = batch["observation.state"].unsqueeze(0)
                if "observation.environment_state" in batch:
                    batch["observation.environment_state"] = batch["observation.environment_state"].unsqueeze(0)

            # 3. Policy select_action() (manages internal temporal action chunk queue)
            with torch.no_grad():
                raw_action = self.policy.select_action(batch)

            # 4. LeRobot Policy Postprocessor (externalized denormalization)
            if self.postprocessor is not None:
                processed_action = self.postprocessor(raw_action)
            else:
                processed_action = raw_action

            # 5. Environment / Action Adapter
            return self.environment_action_adapter(processed_action)

        return proprioception

    def predict_action_chunk(
        self,
        rgb_observation: np.ndarray,
        proprioception: np.ndarray,
        goal_box: list[int] | None = None,
        sub_goal: str = "reach",
        rgb_wrist: np.ndarray | None = None,
        goal_vector: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generates an action chunk of shape (chunk_size, action_dim) at 50Hz.
        Conditioned on multi-modal visual observations and cognitive spatial bounding boxes / goal vector.
        """
        if self.policy is not None:
            actions = []
            for _ in range(self.chunk_size):
                act = self.select_action(rgb_observation, proprioception, rgb_wrist, goal_vector)
                actions.append(act)
            return np.stack(actions, axis=0)

        # Minimum-jerk polynomial trajectory generation toward affordance target
        chunk = np.tile(proprioception, (self.chunk_size, 1))
        t = np.linspace(0, 1.0, self.chunk_size)[:, None]
        # Quintic minimum-jerk polynomial: s(t) = 10*t^3 - 15*t^4 + 6*t^5
        s = 10 * (t**3) - 15 * (t**4) + 6 * (t**5)

        target_delta = np.zeros(self.action_dim)
        if goal_box is not None and len(goal_box) == 4:
            # Map normalized bounding box [ymin, xmin, ymax, xmax] in [0, 1000]
            center_x = (goal_box[1] + goal_box[3]) / 2000.0 - 0.5
            center_y = (goal_box[0] + goal_box[2]) / 2000.0 - 0.5

            sg = sub_goal.lower()
            if sg == "reach":
                target_delta[0] = center_x * 0.9    # Base pan
                target_delta[1] = -center_y * 0.5   # Shoulder lift
                target_delta[2] = 0.25              # Elbow extend
                target_delta[4] = 0.35              # Wrist pitch downward
                target_delta[6] = 0.02              # Gripper open
            elif sg == "grasp":
                target_delta[1] = -center_y * 0.6
                target_delta[2] = 0.32
                target_delta[6] = -0.015            # Gripper close
            elif sg == "lift":
                target_delta[1] = -0.2
                target_delta[2] = 0.1
                target_delta[6] = -0.015            # Maintain grasp
            elif sg == "transport":
                target_delta[0] = center_x * 0.7
                target_delta[1] = -0.3
                target_delta[6] = -0.015            # Maintain grasp
            elif sg == "place":
                target_delta[1] = -0.1
                target_delta[6] = 0.02              # Open gripper
            elif sg in ("retreat", "recover"):
                target_delta[1] = -0.3
                target_delta[2] = 0.1
                target_delta[6] = 0.02              # Gripper open, retreat to clearance
            else:
                target_delta[0] = center_x * 0.7
                target_delta[1] = -0.3
                target_delta[6] = -0.015
        else:
            # Gentle exploratory sweep if no box detected
            target_delta[0] = 0.08 * np.sin(time.time())
            target_delta[1] = -0.15
            target_delta[2] = 0.2

        chunk += s * target_delta
        return chunk
