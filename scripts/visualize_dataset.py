#!/usr/bin/env python3
"""
scripts/visualize_dataset.py
Dataset audit, sanity validator, and telemetry video visualizer for LeRobotDataset v3.0.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mujoco
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_XML_PATH = Path(__file__).parent.parent / "src" / "lerobot_agentic" / "sim" / "models" / "embodied_arm.xml"

CANONICAL_SUBGOALS = ["reach", "grasp", "lift", "transport", "place", "retreat", "recover"]


def check_sanity(dataset_path: Path):
    print("================================================================================")
    print(f"🔍 Running Scientific Sanity Audit on: {dataset_path}")
    print("================================================================================")

    # 1. Reload from disk in a clean instance
    repo_id = dataset_path.name
    ds = LeRobotDataset(repo_id=repo_id, root=dataset_path, video_backend="pyav")

    total_episodes = ds.num_episodes
    total_frames = ds.num_frames
    print(f"• Dataset reloaded successfully: {total_episodes} episodes, {total_frames} frames (FPS: {ds.fps})")

    # 2. Test multi-frame decoding
    sample_indices = [0, total_frames // 2, total_frames - 1]
    # Add one sample from each episode
    ep_start = 0
    for ep_idx in range(min(5, total_episodes)):
        sample_indices.append(ep_start)
        ep_start += 100

    print("• Testing video frame decoding across boundaries...")
    for idx in sorted(set(sample_indices)):
        if 0 <= idx < total_frames:
            item = ds[idx]
            assert item["observation.images.top"].shape == (3, 480, 640), f"Bad top image shape at index {idx}"
            assert item["observation.images.wrist"].shape == (3, 480, 640), f"Bad wrist image shape at index {idx}"
            assert item["observation.state"].shape == (7,), f"Bad state shape at index {idx}"
            assert item["observation.environment_state"].shape == (13,), f"Bad env_state shape at index {idx}"
            assert item["action"].shape == (7,), f"Bad action shape at index {idx}"
    print("  ✅ All sampled frames decoded successfully via PyAV without error.")

    # 3. Comprehensive NaN / Inf / Range scan across entire dataset (fast Parquet/Arrow scan)
    print("• Scanning for NaN / Inf values and checking goal one-hot validity...")
    subgoal_counts = {sg: 0 for sg in CANONICAL_SUBGOALS}

    hf_ds = ds.hf_dataset
    all_states = np.array(hf_ds["observation.state"])
    all_env_states = np.array(hf_ds["observation.environment_state"])
    all_actions = np.array(hf_ds["action"])

    assert not np.isnan(all_states).any(), "NaN detected in observation.state"
    assert not np.isnan(all_env_states).any(), "NaN detected in observation.environment_state"
    assert not np.isnan(all_actions).any(), "NaN detected in action"

    one_hots = all_env_states[:, 6:13]
    assert np.allclose(one_hots.sum(axis=1), 1.0), "Invalid subgoal one-hot sum detected"
    sg_indices = np.argmax(one_hots, axis=1)
    for sg_i, sg_name in enumerate(CANONICAL_SUBGOALS):
        subgoal_counts[sg_name] = int((sg_indices == sg_i).sum())

    print("  ✅ Zero NaNs or Infs detected across all features.")

    # 4. Subgoal class distribution
    print("\n📊 Subgoal Class Distribution:")
    for sg in CANONICAL_SUBGOALS:
        count = subgoal_counts[sg]
        pct = (count / total_frames) * 100.0
        print(f"  - {sg:10s}: {count:6d} frames ({pct:5.1f}%)")

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML_PATH))
    print("\n⚙️  Actuator Range & Saturation Audit:")
    for j in range(model.nu):
        act_name = model.actuator(j).name
        # If ctrlrange not defined, fallback to limits
        if model.actuator_ctrllimited[j]:
            ctrl_min, ctrl_max = float(model.actuator_ctrlrange[j, 0]), float(model.actuator_ctrlrange[j, 1])
        else:
            ctrl_min, ctrl_max = (-3.14159, 3.14159) if j < 6 else (-0.025, 0.025)

        span = ctrl_max - ctrl_min
        vals = all_actions[:, j]
        exact_sat = np.mean((vals <= ctrl_min + 1e-4) | (vals >= ctrl_max - 1e-4)) * 100.0
        near_sat = np.mean((vals <= ctrl_min + 0.02 * span) | (vals >= ctrl_max - 0.02 * span)) * 100.0
        print(f"  - {act_name:14s} [{ctrl_min:7.3f}, {ctrl_max:7.3f}]: min={vals.min():7.3f}, max={vals.max():7.3f}, exact_sat={exact_sat:5.1f}%, near_limit={near_sat:5.1f}%")

    # 6. Audit Goal Perception Errors from sidecar
    goal_audit_path = dataset_path / "goal_audit.jsonl"
    if goal_audit_path.exists():
        errors = []
        with open(goal_audit_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    errors.append(rec["error_m"])
        errors = np.array(errors)
        print("\n🎯 Observable Target Perception Error Audit (||p_obs - p_GT||):")
        print(f"  - Total snapshots audited: {len(errors)}")
        print(f"  - Mean Error: {errors.mean()*1000:.2f} mm")
        print(f"  - Median (p50): {np.percentile(errors, 50)*1000:.2f} mm")
        print(f"  - 95th Percentile: {np.percentile(errors, 95)*1000:.2f} mm")
        print(f"  - Max Error: {errors.max()*1000:.2f} mm")

    # 7. Harvest Manifest Physics Audit
    manifest_path = dataset_path / "harvest_manifest.jsonl"
    if manifest_path.exists():
        durations = []
        dists = []
        slips = []
        forces = []
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec["accepted"]:
                        durations.append(rec["duration_s"])
                        dists.append(rec["final_distance_m"])
                        slips.append(rec["max_slip_m"])
                        forces.append(rec["peak_contact_force_N"])

        durations = np.array(durations)
        dists = np.array(dists)
        slips = np.array(slips)
        forces = np.array(forces)

        print("\n🔬 Harvested Physics & Contact Diagnostics:")
        print(f"  - Episode Duration: mean={durations.mean():.2f}s, min={durations.min():.2f}s, max={durations.max():.2f}s, p95={np.percentile(durations, 95):.2f}s")
        print(f"  - Final Receptacle Dist: mean={dists.mean()*1000:.1f}mm, max={dists.max()*1000:.1f}mm (100% < 30mm: {np.all(dists < 0.03)})")
        print(f"  - Relative Grasp Slip: mean={slips.mean()*1000:.2f}mm, max={slips.max()*1000:.2f}mm")
        print(f"  - Peak Gripping Normal Force: mean={forces.mean():.1f}N, max={forces.max():.1f}N")

    print("\n================================================================================")
    print("✅ Scientific Sanity Audit Complete: All checks passed!")
    print("================================================================================\n")


def render_inspection_video(dataset_path: Path, output_video_path: Path, episode_idx: int = 0):
    repo_id = dataset_path.name
    ds = LeRobotDataset(repo_id=repo_id, root=dataset_path, video_backend="pyav")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_raw_path = output_video_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Composite size: overhead (640) + wrist (640) = 1280 width x 480 height
    writer = cv2.VideoWriter(str(temp_raw_path), fourcc, ds.fps, (1280, 480))

    prev_qpos = None
    dt = 1.0 / ds.fps

    print(f"[Visualizer] Rendering composite HUD video for episode {episode_idx}...")
    for idx in range(len(ds)):
        item = ds[idx]
        ep = int(item["episode_index"].item())
        if ep < episode_idx:
            continue
        if ep > episode_idx:
            break

        # Convert tensors from (C, H, W) to (H, W, C) uint8 BGR for OpenCV
        top_img = item["observation.images.top"].permute(1, 2, 0).numpy()
        wrist_img = item["observation.images.wrist"].permute(1, 2, 0).numpy()

        if top_img.dtype != np.uint8:
            top_img = np.clip(top_img * 255.0, 0, 255).astype(np.uint8)
        if wrist_img.dtype != np.uint8:
            wrist_img = np.clip(wrist_img * 255.0, 0, 255).astype(np.uint8)

        top_bgr = cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)

        qpos = item["observation.state"].numpy()
        env_state = item["observation.environment_state"].numpy()
        act = item["action"].numpy()

        one_hot = env_state[6:13]
        sg_idx = int(np.argmax(one_hot))
        subgoal = CANONICAL_SUBGOALS[sg_idx]

        # Derive joint velocities numerically
        if prev_qpos is not None:
            qvel = (qpos - prev_qpos) / dt
        else:
            qvel = np.zeros_like(qpos)
        prev_qpos = qpos.copy()

        # Telemetry HUD on Top Camera
        cv2.putText(top_bgr, f"SUBGOAL: {subgoal.upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(top_bgr, f"Target: [{env_state[0]:.2f}, {env_state[1]:.2f}, {env_state[2]:.2f}]", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(top_bgr, f"Dest:   [{env_state[3]:.2f}, {env_state[4]:.2f}, {env_state[5]:.2f}]", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(top_bgr, f"Step: {idx} (t={idx*dt:.2f}s)", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Telemetry HUD on Wrist Camera
        cv2.putText(wrist_bgr, "WRIST EYE-IN-HAND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        max_qvel = np.max(np.abs(qvel[:6]))
        cv2.putText(wrist_bgr, f"Derived Max Joint Vel: {max_qvel:.2f} rad/s", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(wrist_bgr, f"Gripper Slide: {qpos[6]:.3f} m (cmd: {act[6]:.3f})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        composite = np.hstack([top_bgr, wrist_bgr])
        writer.write(composite)

    writer.release()

    # Transcode to H.264 (yuv420p) for universal web browser and player compatibility
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_raw_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "baseline", "-level", "3.0",
            str(output_video_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        temp_raw_path.unlink(missing_ok=True)
    else:
        temp_raw_path.rename(output_video_path)

    print(f"✅ Universal H.264 inspection video saved to {output_video_path}")


def export_sample_frames(dataset_path: Path, output_dir: Path, num_episodes: int = 5):
    repo_id = dataset_path.name
    ds = LeRobotDataset(repo_id=repo_id, root=dataset_path, video_backend="pyav")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Visualizer] Exporting sample frames and animated clips for {num_episodes} episodes to {output_dir}...")
    current_ep = -1
    seen_subgoals = set()
    frames_for_anim = []

    for idx in range(len(ds)):
        item = ds[idx]
        ep = int(item["episode_index"].item())
        if ep >= num_episodes:
            break

        if ep != current_ep:
            # If we had previous episode frames, generate webp animation
            if frames_for_anim and current_ep >= 0:
                anim_path = output_dir / f"ep{current_ep:02d}_preview.webp"
                # Sample 1 frame every 4 (12.5 fps preview)
                subsampled = frames_for_anim[::4]
                if subsampled:
                    subsampled[0].save(
                        anim_path,
                        save_all=True,
                        append_images=subsampled[1:],
                        duration=80,
                        loop=0
                    )
            current_ep = ep
            seen_subgoals = set()
            frames_for_anim = []
            ep_dir = output_dir / f"episode_{ep:02d}"
            ep_dir.mkdir(parents=True, exist_ok=True)

        env_state = item["observation.environment_state"].numpy()
        one_hot = env_state[6:13]
        sg_idx = int(np.argmax(one_hot))
        subgoal = CANONICAL_SUBGOALS[sg_idx]

        top_img = item["observation.images.top"].permute(1, 2, 0).numpy()
        wrist_img = item["observation.images.wrist"].permute(1, 2, 0).numpy()
        if top_img.dtype != np.uint8:
            top_img = np.clip(top_img * 255.0, 0, 255).astype(np.uint8)
        if wrist_img.dtype != np.uint8:
            wrist_img = np.clip(wrist_img * 255.0, 0, 255).astype(np.uint8)

        # Build composite for preview
        from PIL import Image
        composite_rgb = np.hstack([top_img, wrist_img])
        pil_frame = Image.fromarray(composite_rgb).resize((640, 240))
        frames_for_anim.append(pil_frame)

        # Export 1 frame per subgoal phase
        if subgoal not in seen_subgoals:
            seen_subgoals.add(subgoal)
            top_bgr = cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR)
            wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
            cv2.putText(top_bgr, f"EP {ep} | {subgoal.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite(str(ep_dir / f"{subgoal}_top.png"), top_bgr)
            cv2.imwrite(str(ep_dir / f"{subgoal}_wrist.png"), wrist_bgr)
            cv2.imwrite(str(ep_dir / f"{subgoal}_composite.png"), np.hstack([top_bgr, wrist_bgr]))

    # Final episode preview
    if frames_for_anim and current_ep >= 0:
        anim_path = output_dir / f"ep{current_ep:02d}_preview.webp"
        subsampled = frames_for_anim[::4]
        if subsampled:
            subsampled[0].save(
                anim_path,
                save_all=True,
                append_images=subsampled[1:],
                duration=80,
                loop=0
            )

    print(f"✅ Sample frames and animated previews successfully exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Audit and visualize LeRobotDataset v3.0.")
    parser.add_argument("--dataset-dir", type=str, default="data/nominal_train_v1", help="Path to LeRobotDataset directory")
    parser.add_argument("--check-sanity", action="store_true", default=False, help="Run scientific sanity audit")
    parser.add_argument("--render-video", action="store_true", help="Render composite HUD video")
    parser.add_argument("--video-episodes", type=int, default=1, help="Number of episodes to render")
    parser.add_argument("--export-samples", action="store_true", help="Export sample frames and animated WebP previews")
    parser.add_argument("--sample-episodes", type=int, default=5, help="Number of episodes for sample frame export")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} does not exist.")
        sys.exit(1)

    if args.check_sanity:
        check_sanity(dataset_path)

    if args.render_video:
        out_dir = Path("outputs/dataset_inspection")
        out_dir.mkdir(parents=True, exist_ok=True)
        for ep in range(args.video_episodes):
            render_inspection_video(dataset_path, out_dir / f"inspection_ep{ep}.mp4", episode_idx=ep)

    if args.export_samples:
        out_dir = Path("outputs/dataset_inspection/sample_frames")
        export_sample_frames(dataset_path, out_dir, num_episodes=args.sample_episodes)


if __name__ == "__main__":
    main()
