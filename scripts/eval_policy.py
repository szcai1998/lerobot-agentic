#!/usr/bin/env python3
"""
scripts/eval_policy.py
Closed-loop simulation rollout evaluator for trained ACT policies (ACT-B and ACT-G).
Benchmarking on fixed held-out validation seeds (2000-2009) in DeepMind MuJoCo.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot_agentic.controllers.ik import ClassicalIKController
from lerobot_agentic.policy.executor import VisuomotorPolicyExecutor
from lerobot_agentic.sim.env import MuJoCoRobotEnv
from lerobot_agentic.utils.metrics import compute_joint_and_gripper_jerk


def compute_wilson_score_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Computes Wilson Score Interval for binomial proportions.
    Particularly suited for small sample sizes (e.g. N=10 validation episodes).
    """
    if total == 0:
        return 0.0, 0.0

    p_hat = successes / total
    # z = 1.95996 for 95% confidence
    z = 1.95996 if math.isclose(confidence, 0.95, rel_tol=1e-2) else 1.96

    denominator = 1.0 + (z**2) / total
    centre_adjusted_probability = p_hat + (z**2) / (2.0 * total)
    adjusted_std_dev = math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)

    lower_bound = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_std_dev) / denominator

    return max(0.0, float(lower_bound)), min(1.0, float(upper_bound))


class ObservableGoalProvider:
    """
    Non-privileged goal snapshot provider for Phase 2 validation of ACT-G.
    Extracts target position via raw overhead RGB-D unprojection (CameraGeometry)
    and canonical subgoal one-hot via deterministic 7-stage Pick-and-Place FSM.
    Canonical vocabulary: reach, grasp, lift, transport, place, retreat, recover.
    Zero privileged state leakage.
    """
    def __init__(self, model, data, receptacle_pos: np.ndarray | None = None):
        self.controller = ClassicalIKController(model, data, receptacle_pos=receptacle_pos)

    def get_goal(self, obs: dict) -> tuple[np.ndarray, str]:
        step = self.controller.act(obs)
        return step.goal_snapshot, step.canonical_subgoal


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ACT policy in closed-loop MuJoCo rollout.")
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to trained policy checkpoint directory (e.g. outputs/checkpoints/act_b_nominal_v1/best_offline).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to benchmark (default: 10).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=2000,
        help="Starting seed for validation episodes (default: 2000).",
    )
    parser.add_argument(
        "--goal-provider",
        type=str,
        default="auto",
        choices=["auto", "rgbd-fsm", "none"],
        help="Goal conditioning vector provider: 'auto' (detects policy feature requirement), 'rgbd-fsm', or 'none'.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=750,
        help="Maximum simulation steps per episode (default: 750 @ 50 Hz = 15.0 s).",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Render and export HUD inspection videos for each rollout.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_rollouts",
        help="Directory to save evaluation summaries and videos (default: outputs/eval_rollouts).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device for policy inference (default: cuda if available).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    policy_path = Path(args.policy_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔍 Closed-Loop Policy Evaluation Benchmark")
    print("=" * 80)
    print(f"• Checkpoint Path:   {policy_path}")
    print(f"• Evaluation Seeds:  {args.seed_start} to {args.seed_start + args.episodes - 1} ({args.episodes} episodes)")
    print(f"• Device:            {args.device}")

    # 1. Instantiate Policy Executor
    executor = VisuomotorPolicyExecutor(
        pretrained_policy_path=str(policy_path),
        chunk_size=50,
        n_action_steps=10,
        device=args.device,
    )

    # Detect whether policy expects environment_state
    expects_goal = (
        executor.policy is not None
        and hasattr(executor.policy.config, "input_features")
        and "observation.environment_state" in executor.policy.config.input_features
    )

    if args.goal_provider == "auto":
        use_goal_provider = expects_goal
    elif args.goal_provider == "rgbd-fsm":
        use_goal_provider = True
    else:
        use_goal_provider = False

    print(f"• Policy Type:       {'ACT-G (Goal-Conditioned)' if expects_goal else 'ACT-B (Unconditioned Baseline)'}")
    print(f"• Goal Provider:     {'ObservableGoalProvider (RGB-D Unprojection FSM)' if use_goal_provider else 'None'}")

    env = MuJoCoRobotEnv()
    dest_pos = np.array([0.295, -0.140, 0.44], dtype=np.float32)

    results = []
    successes = 0

    print("\n🚀 Executing Rollouts...")
    for ep_idx in range(args.episodes):
        seed = args.seed_start + ep_idx
        _obs = env.reset(seed=seed, include_depth=True)

        # Invariant 1: Flush any residual action chunk queue at episode reset
        initial_reset_count = executor.reset_count
        executor.reset()
        assert executor.reset_count == initial_reset_count + 1, "executor.reset() must increment reset_count"
        if hasattr(executor.policy, "_action_queue"):
            assert len(executor.policy._action_queue) == 0, "Action queue must be empty after policy.reset()"

        initial_inferences = executor.inference_count
        initial_control_steps = executor.control_steps
        visited_subgoals: list[str] = []

        goal_provider = ObservableGoalProvider(env.model, env.data, receptacle_pos=dest_pos) if use_goal_provider else None

        joint_history = []
        video_frames = []

        relative_offset_at_grasp = None
        max_grasp_slip = 0.0
        peak_force = 0.0

        step_t = 0
        terminal_reason = "timeout"

        for step_t in range(args.max_steps):
            obs_current = env.get_observation(include_depth=True)
            joint_history.append(obs_current["proprioception"][:7].copy())

            # Obtain non-privileged goal snapshot and canonical subgoal if required
            if goal_provider is not None:
                goal_vector, canonical_subgoal = goal_provider.get_goal(obs_current)
                if not visited_subgoals or visited_subgoals[-1] != canonical_subgoal:
                    visited_subgoals.append(canonical_subgoal)
            else:
                goal_vector = None
                canonical_subgoal = None

            # Policy inference & receding-horizon step
            action = executor.select_action(
                rgb_top=obs_current["rgb"],
                proprioception=obs_current["proprioception"],
                rgb_wrist=obs_current["rgb_wrist"],
                goal_vector=goal_vector,
            )

            # Invariant 2: Actions must be finite
            assert np.all(np.isfinite(action)), f"Non-finite action emitted at step {step_t}: {action}"

            # Invariant 3: Actions must respect clipped actuator limits
            assert np.all(action[:6] >= -3.15) and np.all(action[:6] <= 3.15), f"Arm action out of bounds: {action[:6]}"
            assert -0.026 <= action[6] <= 0.026, f"Gripper action out of bounds: {action[6]}"

            # Invariant 4: Conditioning feature assertions
            if expects_goal:
                assert executor.last_env_state_shape == (13,), f"ACT-G must receive exactly (13,) environment_state, got {executor.last_env_state_shape}"
            else:
                assert "observation.environment_state" not in executor.last_batch_keys, f"ACT-B must receive no environment_state, got keys: {executor.last_batch_keys}"

            # Step physical environment
            _next_obs, _reward, done, _info = env.step(action)

            # Physics telemetry tracking
            cube_pos = env.get_cube_position()
            ee_pos = env.get_ee_position()

            if cube_pos[2] > 0.45 and relative_offset_at_grasp is None:
                ee_mat = env.data.site_xmat[env.ee_site_id].reshape(3, 3) if env.ee_site_id != -1 else np.eye(3)
                relative_offset_at_grasp = ee_mat.T @ (cube_pos - ee_pos)

            diag = env.get_contact_diagnostics(
                grasp_active=bool(cube_pos[2] > 0.45),
                grasp_ref_relative_pos=relative_offset_at_grasp,
            )
            normal_force = float(diag["finger_contact_force_normal"])
            peak_force = max(peak_force, normal_force)
            slip = float(diag["relative_slip"])
            max_grasp_slip = max(max_grasp_slip, slip)

            if args.render_video:
                hud_frame = obs_current["rgb"].copy()
                hud_frame = cv2.cvtColor(hud_frame, cv2.COLOR_RGB2BGR)
                sg_text = f" | SG: {canonical_subgoal}" if canonical_subgoal else ""
                cv2.putText(
                    hud_frame,
                    f"Ep {ep_idx} | Seed {seed} | Step {step_t:3d}{sg_text} | Slip: {max_grasp_slip*1000:.1f}mm",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                video_frames.append(hud_frame)

            if done:
                terminal_reason = "env_terminal"
                break

        # Invariant 5: No fallback action path used
        assert executor.fallback_count == 0, f"Fallback action path was used {executor.fallback_count} times!"

        ep_control_steps = executor.control_steps - initial_control_steps
        ep_inferences = executor.inference_count - initial_inferences
        step_inference_ratio = ep_control_steps / max(1, ep_inferences)

        # Final placement distance evaluation
        final_cube_pos = env.get_cube_position()
        dist_to_dest = float(np.linalg.norm(final_cube_pos - dest_pos))
        is_success = bool(dist_to_dest <= 0.030)  # 30.0 mm physical receptacle gate

        if is_success:
            successes += 1

        # Trajectory jerk computation (Separated: Arm rad/s^3, Gripper m/s^3)
        jerk_dict = compute_joint_and_gripper_jerk(np.array(joint_history), dt=0.02)
        arm_jerk = jerk_dict["arm_joint_jerk_rms_rad_s3"]
        gripper_jerk = jerk_dict["gripper_jerk_rms_m_s3"]

        ep_result = {
            "episode": ep_idx,
            "seed": seed,
            "steps": step_t + 1,
            "duration_s": round((step_t + 1) * 0.02, 2),
            "success": is_success,
            "final_distance_mm": round(dist_to_dest * 1000.0, 2),
            "max_slip_mm": round(max_grasp_slip * 1000.0, 2),
            "peak_force_N": round(peak_force, 2),
            "arm_joint_jerk_rms_rad_s3": arm_jerk,
            "gripper_jerk_rms_m_s3": gripper_jerk,
            "control_steps": ep_control_steps,
            "inferences": ep_inferences,
            "control_step_inference_ratio": round(step_inference_ratio, 2),
            "subgoals_traversed": visited_subgoals,
            "terminal_reason": terminal_reason,
        }
        results.append(ep_result)

        # Video export
        if args.render_video and video_frames:
            vid_path = output_dir / f"rollout_ep{ep_idx}_seed{seed}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w, _ = video_frames[0].shape
            out = cv2.VideoWriter(str(vid_path), fourcc, 50.0, (w, h))
            for f in video_frames:
                out.write(f)
            out.release()

        status_sym = "✅ PASS" if is_success else "❌ FAIL"
        sg_str = f" | Subgoals: {','.join(visited_subgoals)}" if visited_subgoals else ""
        print(
            f"[{ep_idx+1:2d}/{args.episodes:2d}] Seed {seed} | {status_sym} | "
            f"Steps: {step_t+1:3d} (Inferences: {ep_inferences:2d}, Ratio: {step_inference_ratio:4.1f}:1){sg_str} | "
            f"Dist: {dist_to_dest*1000.0:5.1f}mm | Arm Jerk: {arm_jerk:5.1f} rad/s^3 | Grip Jerk: {gripper_jerk:.4f} m/s^3"
        )

    # Summary Statistics & Wilson Score CI
    success_rate = (successes / args.episodes) * 100.0
    ci_low, ci_high = compute_wilson_score_ci(successes, args.episodes, confidence=0.95)

    distances = [r["final_distance_mm"] for r in results]
    slips = [r["max_slip_mm"] for r in results]
    arm_jerks = [r["arm_joint_jerk_rms_rad_s3"] for r in results]
    gripper_jerks = [r["gripper_jerk_rms_m_s3"] for r in results]

    summary = {
        "policy_path": str(policy_path),
        "policy_type": "act-g" if expects_goal else "act-b",
        "total_episodes": args.episodes,
        "successes": successes,
        "success_rate_pct": round(success_rate, 2),
        "wilson_ci_95": [round(ci_low * 100.0, 2), round(ci_high * 100.0, 2)],
        "final_distance_mm": {
            "mean": round(float(np.mean(distances)), 2),
            "std": round(float(np.std(distances)), 2),
            "min": round(float(np.min(distances)), 2),
            "max": round(float(np.max(distances)), 2),
        },
        "max_slip_mm": {
            "mean": round(float(np.mean(slips)), 2),
            "max": round(float(np.max(slips)), 2),
        },
        "arm_joint_jerk_rms_rad_s3": {
            "mean": round(float(np.mean(arm_jerks)), 2),
            "max": round(float(np.max(arm_jerks)), 2),
        },
        "gripper_jerk_rms_m_s3": {
            "mean": round(float(np.mean(gripper_jerks)), 4),
            "max": round(float(np.max(gripper_jerks)), 4),
        },
        "inference_telemetry": {
            "mean_control_to_inference_ratio": round(float(np.mean([r["control_step_inference_ratio"] for r in results])), 2),
            "total_control_steps": sum(r["control_steps"] for r in results),
            "total_inferences": sum(r["inferences"] for r in results),
        },
        "episodes": results,
    }

    summary_file = output_dir / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("📊 Evaluation Benchmark Summary:")
    print("=" * 80)
    print(f"• Success Rate:             {success_rate:.1f}% (Wilson 95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%])")
    print(f"• Mean Placement Distance:  {np.mean(distances):.2f} mm (Max: {np.max(distances):.2f} mm)")
    print(f"• Mean Relative Grasp Slip: {np.mean(slips):.2f} mm (Max: {np.max(slips):.2f} mm)")
    print(f"• Mean Arm Joint Jerk:      {np.mean(arm_jerks):.2f} rad/s^3")
    print(f"• Mean Gripper Jerk:        {np.mean(gripper_jerks):.4f} m/s^3")
    print(f"• Mean Ctrl/Inf Ratio:      {summary['inference_telemetry']['mean_control_to_inference_ratio']:.1f}:1")
    print(f"• Summary JSON:             {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
