import numpy as np

from lerobot_agentic.policy.executor import VisuomotorPolicyExecutor
from lerobot_agentic.sim.env import MuJoCoRobotEnv


def test_mujoco_env_lifecycle():
    env = MuJoCoRobotEnv()
    obs = env.reset()
    assert "rgb" in obs
    assert "rgb_wrist" in obs
    assert "proprioception" in obs
    assert "ee_pos" in obs
    assert obs["rgb"].shape == (480, 640, 3)
    assert obs["rgb_wrist"].shape == (480, 640, 3)
    assert len(obs["proprioception"]) == 7
    assert len(obs["ee_pos"]) == 3

    target_action = np.zeros(7, dtype=np.float32)
    next_obs, _reward, _terminated, info = env.step(target_action)
    assert next_obs["rgb"].shape == (480, 640, 3)
    assert next_obs["rgb_wrist"].shape == (480, 640, 3)
    assert "distance_to_cube" in info

def test_policy_executor_chunking():
    executor = VisuomotorPolicyExecutor(chunk_size=50, action_dim=7)
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_proprio = np.zeros(7, dtype=np.float32)
    chunk = executor.predict_action_chunk(dummy_rgb, dummy_proprio, goal_box=[400, 400, 600, 600])
    assert chunk.shape == (50, 7)
    executor.reset()  # Flushes internal queue state

def test_lerobot_and_cuda_harness():
    """Verifies Stage 1 prerequisites: LeRobot import, PyTorch CUDA GPU, and MuJoCo headless context."""
    import lerobot
    import mujoco
    import torch

    assert torch.cuda.is_available(), "CUDA acceleration must be operational for RTX 3070"
    assert hasattr(lerobot, "__version__"), "LeRobot must expose __version__"
    assert mujoco.MjModel is not None, "MuJoCo library must be loaded"


def test_goal_vector_and_unbatched_processor():
    """Verifies 13-DoF goal vector encoding and unbatched observation tensors."""
    from lerobot_agentic.policy.executor import encode_goal_vector

    # 13-DoF: 3 (target) + 3 (dest) + 7 (one-hot subgoal)
    goal_vec = encode_goal_vector(
        target_pos=[0.32, 0.05, 0.43],
        dest_pos=[0.32, -0.15, 0.40],
        sub_goal="place"
    )
    assert goal_vec.shape == (13,)
    assert goal_vec[6 + 4] == 1.0  # 'place' is index 4 in canonical subgoals

    executor = VisuomotorPolicyExecutor(chunk_size=50, action_dim=7)
    dummy_top = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_wrist = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_state = np.zeros(7, dtype=np.float32)

    # Environment processor must produce UNBATCHED tensors
    batch = executor.environment_processor(
        dummy_top,
        dummy_state,
        rgb_wrist=dummy_wrist,
        goal_vector=goal_vec
    )
    assert batch["observation.images.top"].shape == (3, 480, 640), "Top image must be (C, H, W) unbatched"
    assert batch["observation.images.wrist"].shape == (3, 480, 640), "Wrist image must be (C, H, W) unbatched"
    assert batch["observation.state"].shape == (7,), "State must be (7,) unbatched"
    assert batch["observation.environment_state"].shape == (13,), "Environment state must be (13,) unbatched"
