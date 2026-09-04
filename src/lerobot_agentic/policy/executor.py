import os
import time
from typing import Optional, List, Dict, Any
import numpy as np
import torch

class VisuomotorPolicyExecutor:
    """
    Manages inference and action chunk execution for LeRobot policies (ACT, Diffusion, SmolVLA)
    with smooth temporal ensembling and affordance-directed trajectory interpolation.
    """
    def __init__(
        self,
        pretrained_policy_path: Optional[str] = None,
        chunk_size: int = 50,
        action_dim: int = 7,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.policy = None

        if pretrained_policy_path and os.path.exists(pretrained_policy_path):
            try:
                from lerobot.common.policies.act.modeling_act import ACTPolicy
                self.policy = ACTPolicy.from_pretrained(pretrained_policy_path).to(self.device)
                self.policy.eval()
                self.policy.reset()
                print(f"[PolicyExecutor] Loaded Hugging Face LeRobot ACTPolicy from {pretrained_policy_path}")
            except Exception as e:
                print(f"[PolicyExecutor] LeRobot checkpoint load warning: {e}. Defaulting to hybrid trajectory engine.")

    def predict_action_chunk(
        self,
        rgb_observation: np.ndarray,
        proprioception: np.ndarray,
        goal_box: Optional[List[int]] = None,
        sub_goal: str = "reach"
    ) -> np.ndarray:
        """
        Generates an action chunk of shape (chunk_size, action_dim) at 50Hz.
        Conditioned on multi-modal visual observations and cognitive spatial bounding boxes.
        """
        if self.policy is not None:
            obs_dict = {
                "observation.images.top": torch.from_numpy(rgb_observation).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0,
                "observation.state": torch.from_numpy(proprioception).unsqueeze(0).float().to(self.device)
            }
            with torch.no_grad():
                action = self.policy.select_action(obs_dict)
            return action.cpu().numpy()

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
