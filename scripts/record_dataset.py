#!/usr/bin/env python3
"""
scripts/record_dataset.py
Harvester CLI to collect expert pick-and-place demonstration episodes into
the Hugging Face LeRobotDataset v3.0 standard.

Guarantees:
- Zero privileged-state leakage into ACT conditioning (observable RGB-D unprojection).
- Atomic (o_t, g_t, a_t) step alignment (no 1-step label shift).
- Full auditability via harvest_manifest.jsonl and goal_audit.jsonl.
- Complete hash manifests: dataset_manifest.json & dataset_manifest.sha256.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import av
import lerobot
import mujoco
import torch
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_agentic.controllers.ik import ClassicalIKController
from lerobot_agentic.sim.env import MuJoCoRobotEnv


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Record expert demonstrations into LeRobotDataset v3.0 format.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of accepted episodes to record")
    parser.add_argument("--base-seed", type=int, default=1000, help="Starting deterministic seed for episode generation")
    parser.add_argument("--output-dir", type=str, default="data/nominal_train_v1", help="Output directory for LeRobotDataset")
    parser.add_argument("--scenario", type=str, default="nominal", choices=["nominal", "recovery"], help="Demonstration scenario")
    parser.add_argument("--fps", type=int, default=50, help="Control and video frame rate")
    parser.add_argument("--max-steps", type=int, default=450, help="Maximum steps per episode before timeout")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    if output_path.exists():
        import shutil
        print(f"[Harvester] Removing existing dataset directory: {output_path}")
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repo_id = output_path.name

    print("================================================================================")
    print(f"🦾 LeRobot-Agentic Demonstration Harvester (LeRobot {lerobot.__version__})")
    print(f"Target Directory: {output_path}")
    print(f"Episodes: {args.episodes} | Base Seed: {args.base_seed} | Scenario: {args.scenario}")
    print("================================================================================")

    features = {
        "observation.images.top": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
        "observation.environment_state": {"dtype": "float32", "shape": (13,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }

    rgb_encoder = RGBEncoderConfig(vcodec="h264", video_backend="pyav")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        features=features,
        root=output_path,
        use_videos=True,
        rgb_encoder=rgb_encoder,
        video_backend="pyav",
    )

    env = MuJoCoRobotEnv()
    ctrl = ClassicalIKController(env.model, env.data)

    manifest_path = output_path / "harvest_manifest.jsonl"
    goal_audit_path = output_path / "goal_audit.jsonl"
    manifest_records = []
    goal_audit_records = []

    accepted_episodes = 0
    candidate_seed = args.base_seed
    attempt_idx = 0

    while accepted_episodes < args.episodes:
        seed = candidate_seed
        candidate_seed += 1
        attempt_idx += 1

        env.reset(seed=seed, include_depth=True)
        ctrl.reset_state_machine()

        peak_force = 0.0
        max_penetration = 0.0
        max_slip = 0.0
        episode_goal_audits = []

        step_count = 0
        while step_count < args.max_steps:
            obs_t = env.get_observation(include_depth=True)
            step = ctrl.act(obs_t)

            # Record frame in LeRobot buffer
            frame_data = {
                "observation.images.top": obs_t["rgb"],
                "observation.images.wrist": obs_t["rgb_wrist"],
                "observation.state": obs_t["proprioception"],
                "observation.environment_state": step.goal_snapshot,
                "action": step.action,
                "task": "Pick the red cube and place it in the green receptacle",
            }
            dataset.add_frame(frame_data)

            # Physical contact diagnostics
            diag = env.get_contact_diagnostics(ctrl.grasp_active, ctrl.grasp_ref_relative_pos)
            peak_force = max(peak_force, diag["finger_contact_force_normal"])
            max_penetration = max(max_penetration, diag["penetration_depth"])
            max_slip = max(max_slip, diag["relative_slip"])

            # Log goal audit record upon canonical transition
            if step.diagnostics["stage_timer"] == 1.0:
                episode_goal_audits.append({
                    "seed": seed,
                    "episode_idx": accepted_episodes,
                    "frame_idx": step_count,
                    "canonical_subgoal": step.canonical_subgoal,
                    "observed_target_xyz": step.goal_snapshot[:3].tolist(),
                    "gt_target_xyz": ctrl.get_gt_target().tolist(),
                    "error_m": step.diagnostics["perception_error_m"],
                })

            # Step physics
            _obs, _reward, _done, _info = env.step(step.action)
            step_count += 1

            if step.controller_terminal:
                break

        # Settle window (allow physics to damp residual motion)
        for _ in range(10):
            settle_act = np.concatenate([env.get_proprioception()[:6], [0.025]])
            env.step(settle_act)

        cube_final = env.get_cube_position()
        dist_to_rec = float(np.linalg.norm(cube_final[:2] - np.array([0.32, -0.15])))
        is_success = env.check_settled_task_success()

        failure_reason = None
        if not is_success:
            if dist_to_rec >= 0.03:
                failure_reason = f"placement_distance_exceeded_{dist_to_rec*1000:.1f}mm"
            else:
                failure_reason = "cube_unsettled_or_fell"
        elif max_slip > 0.055:
            is_success = False
            failure_reason = f"excessive_grasp_slip_{max_slip*1000:.1f}mm"
        elif max_penetration > 0.02:
            is_success = False
            failure_reason = f"excessive_penetration_{max_penetration*1000:.1f}mm"
        elif peak_force > 30.0:
            is_success = False
            failure_reason = f"excessive_force_{peak_force:.1f}N"

        attempt_record = {
            "seed": seed,
            "episode_idx": accepted_episodes if is_success else None,
            "attempt_idx": attempt_idx,
            "accepted": is_success,
            "failure_reason": failure_reason,
            "steps": step_count,
            "duration_s": round(step_count * 0.02, 3),
            "final_distance_m": round(dist_to_rec, 5),
            "peak_contact_force_N": round(peak_force, 2),
            "max_penetration_m": round(max_penetration, 5),
            "max_slip_m": round(max_slip, 5),
            "git_commit": get_git_commit(),
        }
        manifest_records.append(attempt_record)

        if is_success:
            dataset.save_episode()
            goal_audit_records.extend(episode_goal_audits)
            accepted_episodes += 1
            print(f"[Episode {accepted_episodes:02d}/{args.episodes}] Seed {seed} (Attempt {attempt_idx}) | Steps: {step_count} ({step_count*0.02:.1f}s) | Dist: {dist_to_rec*1000:.1f}mm | Slip: {max_slip*1000:.2f}mm | Force: {peak_force:.1f}N ✅")
        else:
            dataset.clear_episode_buffer()
            print(f"[REJECTED] Seed {seed} (Attempt {attempt_idx}) | Reason: {failure_reason} ❌")

    print("\n[Harvester] Invoking dataset.finalize() to flush video/Parquet writers and compute stats...")
    dataset.finalize()
    print("✅ dataset.finalize() completed!")

    # Write harvest manifest
    with open(manifest_path, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in manifest_records)

    # Write goal audit sidecar
    with open(goal_audit_path, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in goal_audit_records)

    # Generate immutable dataset_manifest.json
    file_hashes = {}
    for root, _, files in os.walk(output_path):
        for file in sorted(files):
            if file in ("dataset_manifest.json", "dataset_manifest.sha256"):
                continue
            fpath = Path(root) / file
            rel_path = str(fpath.relative_to(output_path))
            file_hashes[rel_path] = compute_file_sha256(fpath)

    uv_lock_path = Path(__file__).parent.parent / "uv.lock"
    uv_lock_sha = compute_file_sha256(uv_lock_path) if uv_lock_path.exists() else "unknown"

    dataset_manifest = {
        "dataset_version": repo_id,
        "scenario": args.scenario,
        "git_commit": get_git_commit(),
        "uv_lock_sha256": uv_lock_sha,
        "environment": {
            "python": sys.version.split()[0],
            "lerobot": lerobot.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "av": av.__version__,
        },
        "seed_partition": {
            "base_seed": args.base_seed,
            "end_seed": args.base_seed + args.episodes - 1,
            "total_accepted_seeds": args.episodes,
        },
        "dataset_stats": {
            "num_episodes": dataset.meta.total_episodes,
            "num_frames": dataset.meta.total_frames,
            "fps": args.fps,
        },
        "file_sha256": file_hashes,
    }

    manifest_json_path = output_path / "dataset_manifest.json"
    with open(manifest_json_path, "w") as f:
        json.dump(dataset_manifest, f, indent=2)

    manifest_sha = compute_file_sha256(manifest_json_path)
    with open(output_path / "dataset_manifest.sha256", "w") as f:
        f.write(f"{manifest_sha}  dataset_manifest.json\n")

    print("\n================================================================================")
    print(f"🎉 Harvest Complete: {args.episodes} Accepted Episodes ({dataset.meta.total_frames} total frames)")
    print(f"Manifest written to: {manifest_json_path}")
    print(f"Manifest SHA-256: {manifest_sha}")
    print("================================================================================")


if __name__ == "__main__":
    main()
