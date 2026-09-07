import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_agentic.controllers.ik import ClassicalIKController
from lerobot_agentic.sim.env import MuJoCoRobotEnv


def test_lerobot_dataset_lifecycle_and_reload():
    """
    Verifies that a dataset can be created, populated with frames, finalized,
    closed, and reloaded from disk in a fresh instance with valid PyAV decoding.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        features = {
            "observation.images.top": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
            "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
            "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
            "observation.environment_state": {"dtype": "float32", "shape": (13,), "names": None},
            "action": {"dtype": "float32", "shape": (7,), "names": None},
        }

        rgb_encoder = RGBEncoderConfig(vcodec="h264", video_backend="pyav")
        ds = LeRobotDataset.create(
            repo_id="test_pilot",
            fps=50,
            features=features,
            root=tmp_dir / "test_pilot",
            use_videos=True,
            rgb_encoder=rgb_encoder,
            video_backend="pyav",
        )

        env = MuJoCoRobotEnv()
        _obs = env.reset(seed=1000, include_depth=True)
        ctrl = ClassicalIKController(env.model, env.data)

        # Run 30 steps of demonstration
        for t in range(30):
            obs_t = env.get_observation(include_depth=True)
            step = ctrl.act(obs_t)
            ds.add_frame({
                "observation.images.top": obs_t["rgb"],
                "observation.images.wrist": obs_t["rgb_wrist"],
                "observation.state": obs_t["proprioception"],
                "observation.environment_state": step.goal_snapshot,
                "action": step.action,
                "task": "Pick the red cube and place it in the green receptacle",
            })
            env.step(step.action)

        ds.save_episode()
        ds.finalize()

        # Destroy writer instance and verify reload from disk
        del ds
        reloaded = LeRobotDataset(repo_id="test_pilot", root=tmp_dir / "test_pilot", video_backend="pyav")
        assert reloaded.num_episodes == 1
        assert reloaded.num_frames == 30

        # Test boundary and random decoding
        for idx in [0, 15, 29]:
            frame = reloaded[idx]
            assert frame["observation.images.top"].shape == (3, 480, 640)
            assert frame["observation.images.wrist"].shape == (3, 480, 640)
            assert frame["observation.state"].shape == (7,)
            assert frame["observation.environment_state"].shape == (13,)
            assert frame["action"].shape == (7,)

            one_hot = frame["observation.environment_state"][6:13].numpy()
            assert np.isclose(one_hot.sum(), 1.0)
            assert not torch.isnan(frame["observation.state"]).any()
            assert not torch.isnan(frame["action"]).any()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
