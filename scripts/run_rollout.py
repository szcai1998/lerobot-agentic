#!/usr/bin/env python3
import argparse
import os
import sys
import threading
import time
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot_agentic.cognitive.plan_state import AtomicPlanState, LatestFrameBuffer
from lerobot_agentic.cognitive.supervisor import CognitiveSupervisor
from lerobot_agentic.policy.executor import VisuomotorPolicyExecutor
from lerobot_agentic.sim.env import MuJoCoRobotEnv
from lerobot_agentic.utils.recorder import EpisodeVideoRecorder


def main():
    parser = argparse.ArgumentParser(description="Run LeRobot-Agentic dual-rate closed-loop simulation rollout.")
    parser.add_argument("--goal", type=str, default="Grasp the red cube and lift it into the workspace",
                        help="Natural language task prompt for Cognitive Supervisor")
    parser.add_argument("--steps", type=int, default=150, help="Total 50Hz simulation steps (150 = 3.0s)")
    parser.add_argument("--video", action="store_true", default=True, help="Record MP4 video of rollout")
    parser.add_argument("--gif", action="store_true", help="Record animated GIF of rollout")
    parser.add_argument("--policy-path", type=str, default=None, help="Path to pretrained LeRobot ACTPolicy checkpoint")
    args = parser.parse_args()

    print("================================================================================")
    print("🤖 LeRobot-Agentic: Embodied Manipulation Rollout Engine (Dual-Rate Asynchronous)")
    print(f"Goal: '{args.goal}'")
    print(f"Horizon: {args.steps} steps @ 50 Hz (~{args.steps * 0.02:.1f}s physical time)")
    print("================================================================================")

    env = MuJoCoRobotEnv()
    obs = env.reset()
    policy = VisuomotorPolicyExecutor(pretrained_policy_path=args.policy_path)
    recorder = EpisodeVideoRecorder(output_dir="outputs/videos")
    plan_state = AtomicPlanState()

    # Resolve GEMINI_API_KEY
    if not os.environ.get("GEMINI_API_KEY"):
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if line.startswith("GEMINI_API_KEY="):
                        os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip().strip("\"").strip("\x27")
                        break

    frame_buffer = LatestFrameBuffer()
    frame_buffer.push(obs["rgb"], rgb_wrist=obs.get("rgb_wrist"), step_idx=0)
    stop_event = threading.Event()

    # -------------------------------------------------------------------------
    # Asynchronous Cognitive Supervisor Thread (~0.5 - 2 Hz)
    # -------------------------------------------------------------------------
    def supervisor_worker():
        if not os.environ.get("GEMINI_API_KEY"):
            return
        try:
            supervisor = CognitiveSupervisor()
        except Exception as e:  # noqa: BLE001
            print(f"[Supervisor] Init warning: {e}")
            return

        while not stop_event.is_set():
            rgb_snap, _, _, _ = frame_buffer.get_latest()
            if rgb_snap is not None:
                try:
                    plan = supervisor.plan_and_ground(rgb_snap, args.goal)
                    plan_state.update(plan)
                    print(f"[Supervisor Async] Grounding -> Sub-Goal: '{plan.sub_goal}' | Box: {plan.target_box_2d} | Replan: {plan.requires_replanning}")
                except Exception as e:  # noqa: BLE001
                    print(f"[Supervisor Async Error] {e}")

            time.sleep(0.5)  # Target ~2 Hz cadence, measured empirically

    if os.environ.get("GEMINI_API_KEY"):
        print("[Supervisor] Connected via GEMINI_API_KEY. Launching asynchronous supervisory thread.")
        supervisor_thread = threading.Thread(target=supervisor_worker, daemon=True)
        supervisor_thread.start()
    else:
        print("[Supervisor] GEMINI_API_KEY not set. Running autonomous heuristic tracking.")

    sim_step = 0
    current_chunk = None
    chunk_idx = 0
    consumed_replan_id = 0

    start_time = time.time()

    try:
        while sim_step < args.steps:
            tick_start = time.perf_counter()
            rgb_frame = obs["rgb"]
            rgb_wrist = obs.get("rgb_wrist")
            proprio = obs["proprioception"]

            # Push immutable copy to thread-safe frame buffer
            frame_buffer.push(rgb_frame, rgb_wrist=rgb_wrist, step_idx=sim_step)

            # Read latest plan without blocking 50 Hz execution loop
            active_plan, _target_3d, _dest_3d, _subgoal_idx, _plan_ver = plan_state.get_snapshot()

            # Edge-triggered dynamic closed-loop recovery: reset policy queue only once per anomaly event
            has_new_replan, new_replan_id = plan_state.check_and_consume_replan(consumed_replan_id)
            if has_new_replan:
                consumed_replan_id = new_replan_id
                print(f"[{sim_step * 0.02:4.2f}s] 🔄 [Recovery] Anomaly event {new_replan_id} detected! Resetting LeRobot action queue...")
                policy.reset()
                current_chunk = None

            if active_plan is not None and active_plan.should_halt:
                print("[Safety] Anomaly detected by supervisor. Halting.")
                break

            # Retrieve action chunk (50 steps per chunk = 1s horizon)
            if current_chunk is None or chunk_idx >= len(current_chunk):
                goal_box = active_plan.target_box_2d if (active_plan and active_plan.target_box_2d) else [450, 480, 550, 560]
                sub_goal_label = active_plan.sub_goal if active_plan else ("reach" if sim_step < 75 else "lift")
                current_chunk = policy.predict_action_chunk(
                    rgb_frame,
                    proprio,
                    goal_box=goal_box,
                    sub_goal=sub_goal_label,
                    rgb_wrist=rgb_wrist
                )
                chunk_idx = 0

            target_action = current_chunk[chunk_idx]
            chunk_idx += 1

            # Step physics in MuJoCo at 50 Hz
            obs, _reward, _terminated, info = env.step(target_action)

            # Record video frame with HUD
            if args.video or args.gif:
                current_sub_goal = active_plan.sub_goal if active_plan else ("reach" if sim_step < 75 else "lift")
                box_to_draw = active_plan.target_box_2d if active_plan else [450, 480, 550, 560]
                recorder.add_frame(rgb_frame, sub_goal=current_sub_goal, target_box_2d=box_to_draw, step_idx=sim_step)

            sim_step += 1

            # Optional real-time pacing (50 Hz = 20ms) if pacing enabled
            tick_elapsed = time.perf_counter() - tick_start
            if tick_elapsed < 0.02 and args.policy_path is not None:
                time.sleep(0.02 - tick_elapsed)

    finally:
        stop_event.set()

    elapsed = time.time() - start_time
    fps = sim_step / max(elapsed, 1e-4)
    print(f"\n[Completed] Executed {sim_step} steps in {elapsed:.2f}s ({fps:.1f} FPS).")
    print(f"Final distance to cube: {info['distance_to_cube']:.4f} m | Cube lifted: {info['cube_lifted']}")

    if args.video:
        recorder.save("rollout.mp4")
    if args.gif:
        recorder.save("rollout.gif")


if __name__ == "__main__":
    main()
