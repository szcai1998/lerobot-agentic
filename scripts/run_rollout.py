#!/usr/bin/env python3
import os
import sys
import argparse
import time
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot_agentic.sim.env import MuJoCoRobotEnv
from lerobot_agentic.policy.executor import VisuomotorPolicyExecutor
from lerobot_agentic.cognitive.supervisor import CognitiveSupervisor
from lerobot_agentic.utils.recorder import EpisodeVideoRecorder

def main():
    parser = argparse.ArgumentParser(description="Run LeRobot-Agentic closed-loop simulation rollout.")
    parser.add_argument("--goal", type=str, default="Grasp the red cube and lift it into the workspace",
                        help="Natural language task prompt for Cognitive Supervisor")
    parser.add_argument("--steps", type=int, default=150, help="Total 50Hz simulation steps (150 = 3.0s)")
    parser.add_argument("--video", action="store_true", default=True, help="Record MP4 video of rollout")
    parser.add_argument("--gif", action="store_true", help="Record animated GIF of rollout")
    parser.add_argument("--policy-path", type=str, default=None, help="Path to pretrained LeRobot ACTPolicy checkpoint")
    args = parser.parse_args()

    print("================================================================================")
    print("🤖 LeRobot-Agentic: Embodied Manipulation Rollout Engine")
    print(f"Goal: '{args.goal}'")
    print(f"Horizon: {args.steps} steps @ 50 Hz (~{args.steps * 0.02:.1f}s physical time)")
    print("================================================================================")

    env = MuJoCoRobotEnv()
    obs = env.reset()
    policy = VisuomotorPolicyExecutor(pretrained_policy_path=args.policy_path)
    
    recorder = EpisodeVideoRecorder(output_dir="outputs/videos")

    supervisor = None
    if os.environ.get("GEMINI_API_KEY"):
        print("[Supervisor] Connected via GEMINI_API_KEY. Using Gemini Robotics ER.")
        try:
            supervisor = CognitiveSupervisor()
        except Exception as e:
            print(f"[Supervisor] Init warning: {e}. Falling back to autonomous heuristic.")
    else:
        print("[Supervisor] GEMINI_API_KEY not set. Running autonomous affordance tracking.")

    sim_step = 0
    current_chunk = None
    chunk_idx = 0
    active_plan = None

    start_time = time.time()
    
    while sim_step < args.steps:
        rgb_frame = obs["rgb"]
        proprio = obs["proprioception"]

        # Run cognitive supervisor every 50 steps (1 Hz) or at start
        if sim_step % 50 == 0:
            if supervisor is not None:
                try:
                    active_plan = supervisor.plan_and_ground(rgb_frame, args.goal)
                    print(f"[{sim_step * 0.02:4.2f}s] Gemini Grounding -> Sub-Goal: '{active_plan.sub_goal}' | Box: {active_plan.target_box_2d}")
                    if active_plan.should_halt:
                        print("[Safety] Anomaly detected by supervisor. Halting.")
                        break
                except Exception as e:
                    print(f"[{sim_step * 0.02:4.2f}s] Supervisor error: {e}")
            else:
                # Default pseudo-grounding for target cube
                active_plan_sub_goal = "reach_cube" if sim_step < 75 else "grasp_and_lift"
                active_box = [450, 480, 550, 560]  # Centered bounding box in normalized space

        # Retrieve action chunk (50 steps per chunk = 1s horizon)
        if current_chunk is None or chunk_idx >= len(current_chunk):
            goal_box = active_plan.target_box_2d if (active_plan and active_plan.target_box_2d) else [450, 480, 550, 560]
            sub_goal_label = active_plan.sub_goal if active_plan else ("reach" if sim_step < 75 else "lift")
            current_chunk = policy.predict_action_chunk(rgb_frame, proprio, goal_box=goal_box, sub_goal=sub_goal_label)
            chunk_idx = 0

        target_action = current_chunk[chunk_idx]
        chunk_idx += 1

        # Step physics in MuJoCo
        obs, reward, terminated, info = env.step(target_action)
        
        # Record video frame with HUD
        if args.video or args.gif:
            current_sub_goal = active_plan.sub_goal if active_plan else ("reach" if sim_step < 75 else "lift")
            box_to_draw = active_plan.target_box_2d if active_plan else [450, 480, 550, 560]
            recorder.add_frame(rgb_frame, sub_goal=current_sub_goal, target_box_2d=box_to_draw, step_idx=sim_step)

        sim_step += 1

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
