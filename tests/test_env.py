import pytest
import numpy as np
from lerobot_agentic.sim.env import MuJoCoRobotEnv
from lerobot_agentic.policy.executor import VisuomotorPolicyExecutor

def test_mujoco_env_lifecycle():
    env = MuJoCoRobotEnv()
    obs = env.reset()
    assert "rgb" in obs
    assert "proprioception" in obs
    assert obs["rgb"].shape == (480, 640, 3)
    assert len(obs["proprioception"]) == 7

    target_action = np.zeros(7, dtype=np.float32)
    next_obs, reward, terminated, info = env.step(target_action)
    assert next_obs["rgb"].shape == (480, 640, 3)
    assert "distance_to_cube" in info

def test_policy_executor_chunking():
    executor = VisuomotorPolicyExecutor(chunk_size=50, action_dim=7)
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_proprio = np.zeros(7, dtype=np.float32)
    chunk = executor.predict_action_chunk(dummy_rgb, dummy_proprio, goal_box=[400, 400, 600, 600])
    assert chunk.shape == (50, 7)

def test_lerobot_and_cuda_harness():
    """Verifies Stage 1 prerequisites: LeRobot import, PyTorch CUDA GPU, and MuJoCo headless context."""
    import torch
    import lerobot
    import mujoco

    assert torch.cuda.is_available(), "CUDA acceleration must be operational for RTX 3070"
    assert hasattr(lerobot, "__version__"), "LeRobot must expose __version__"
    assert mujoco.MjModel is not None, "MuJoCo library must be loaded"
