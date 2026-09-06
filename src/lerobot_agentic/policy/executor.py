import os
import time
from typing import Any

import numpy as np
import torch


class VisuomotorPolicyExecutor:
    """
    Manages inference and action chunk execution for LeRobot policies (ACT, Diffusion, SmolVLA)
    with smooth temporal ensembling, PolicyProcessorPipeline integration, and affordance-directed
    trajectory interpolation.
    """
    def __init__(
        self,
        pretrained_policy_path: str | None = None,
        chunk_size: int = 50,
        action_dim: int = 7,
        device: str | None = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None

        if pretrained_policy_path and os.path.exists(pretrained_policy_path):
            try:
                from lerobot.common.policies.act.modeling_act import ACTPolicy
                self.policy = ACTPolicy.from_pretrained(pretrained_policy_path).to(self.device)
                self.policy.eval()
                self.policy.reset()

                # LeRobot 0.6+ PolicyProcessorPipeline initialization
                try:
                    from lerobot.policies import make_pre_post_processors
                    self.preprocessor, self.postprocessor = make_pre_post_processors(
                        policy_cfg=self.policy.config
                    )
                except Exception:  # noqa: BLE001, S110
                    # Preprocessor pipeline optional depending on checkpoint metadata
                    pass

                print(f"[PolicyExecutor] Loaded Hugging Face LeRobot ACTPolicy from {pretrained_policy_path}")
            except Exception as e:  # noqa: BLE001
                print(f"[PolicyExecutor] LeRobot checkpoint load warning: {e}. Defaulting to hybrid trajectory engine.")

    def reset(self):
        """Flushes any cached action chunk queue in the policy during replanning."""
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def environment_processor(
        self,
        rgb_top: np.ndarray,
        proprioception: np.ndarray,
        rgb_wrist: np.ndarray | None = None,
        goal_vector: np.ndarray | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Stage 1: Raw MuJoCo observation -> Environment Processor.
        Converts sensor arrays into raw tensor dictionary adhering to LeRobot dataset keys.
        """
        batch = {
            "observation.images.top": torch.from_numpy(rgb_top).permute(2, 0, 1).unsqueeze(0).to(self.device),
            "observation.state": torch.from_numpy(proprioception).unsqueeze(0).float().to(self.device)
        }
        if rgb_wrist is not None:
            batch["observation.images.wrist"] = torch.from_numpy(rgb_wrist).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if goal_vector is not None:
            batch["observation.goal"] = torch.from_numpy(goal_vector).unsqueeze(0).float().to(self.device)
        return batch

    def environment_action_adapter(self, action: Any) -> np.ndarray:
        """
        Stage 5: Environment / Action Adapter -> MuJoCo actuator command.
        Converts postprocessed tensor to numpy joint command and clips to actuator limits.
        """
        if isinstance(action, torch.Tensor):
            action_np = action.squeeze(0).detach().cpu().numpy()
        else:
            action_np = np.asarray(action)
        return np.clip(action_np, -3.14, 3.14)

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

            # 2. LeRobot Policy Preprocessor (externalized normalization)
            if self.preprocessor is not None:
                batch = self.preprocessor(batch)
            else:
                batch["observation.images.top"] = batch["observation.images.top"].float() / 255.0
                if "observation.images.wrist" in batch:
                    batch["observation.images.wrist"] = batch["observation.images.wrist"].float() / 255.0

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
        goal_vector: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Generates an action chunk of shape (chunk_size, action_dim) at 50Hz.
        Conditioned on multi-modal visual observations and cognitive spatial bounding boxes / goal vector.
        """
        if self.policy is not None:
            # Step the policy chunk_size times to build a chunk
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

            if "reach" in sub_goal.lower():
                target_delta[0] = center_x * 0.9    # Base pan
                target_delta[1] = -center_y * 0.5   # Shoulder lift
                target_delta[2] = 0.25              # Elbow extend
                target_delta[4] = 0.35              # Wrist pitch downward
                target_delta[6] = 0.02              # Gripper open
            elif "grasp" in sub_goal.lower():
                target_delta[1] = -center_y * 0.6
                target_delta[2] = 0.32
                target_delta[6] = -0.015            # Gripper close
            elif "lift" in sub_goal.lower():
                target_delta[1] = -0.2
                target_delta[2] = 0.1
                target_delta[6] = -0.015            # Maintain grasp
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
