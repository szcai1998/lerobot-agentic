#!/usr/bin/env python3
"""
scripts/run_production_pipeline.py
Autonomous end-to-end orchestrator for Phase 2 Part 2:
1. Monitors & waits for currently running ACT-B training to complete (10,000 steps).
2. Launches ACT-G 10,000-step training with identical hyperparameters & observable goal conditioning.
3. Benchmarks both ACT-B and ACT-G checkpoints in closed-loop MuJoCo rollouts across held-out seeds (2000-2009).
4. Generates a consolidated benchmark comparison report and emits outputs/PIPELINE_COMPLETE.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def log_phase(title: str):
    separator = "=" * 80
    print(f"\n{separator}\n🔔 {title}\n{separator}\n", flush=True)


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_running_training_job(policy_type: str) -> int | None:
    try:
        output = subprocess.check_output(
            ["pgrep", "-f", f"scripts/train_policy.py.*--policy-type {policy_type}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.strip().splitlines():
            parts = line.split()
            if parts:
                pid = int(parts[0])
                if pid != os.getpid():
                    comm_file = Path(f"/proc/{pid}/comm")
                    if comm_file.is_file() and "python" in comm_file.read_text().lower():
                        return pid
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def _stream_proc(proc: subprocess.Popen, f_out) -> int:
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if f_out is not None:
            f_out.write(line)
            f_out.flush()
    proc.wait()
    return proc.returncode


def run_command_stream(cmd: list[str], log_file: Path | None = None) -> int:
    print(f"▶ Executing: {' '.join(cmd)}", flush=True)
    if log_file:
        with open(log_file, "w") as f_out:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            return _stream_proc(proc, f_out)
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        return _stream_proc(proc, None)


def wait_for_act_b(output_dir: Path, check_interval: int = 20):
    log_phase("Stage 1/3: Monitoring Active ACT-B Training")
    best_checkpoint = output_dir / "best_offline"

    pid = find_running_training_job("act-b")
    if pid is not None:
        print(f"• Detected active ACT-B training process PID: {pid}", flush=True)
        while is_process_running(pid):
            telem_file = output_dir / "training_telemetry.jsonl"
            if telem_file.is_file():
                try:
                    last_line = subprocess.check_output(["tail", "-n", "1", str(telem_file)], text=True).strip()
                    if last_line:
                        entry = json.loads(last_line)
                        step = entry.get("step", "?")
                        l1 = entry.get("train_l1", entry.get("val_prior_l1", "?"))
                        vram = entry.get("peak_vram_mb", "?")
                        print(f"  [ACT-B Monitoring] Step: {step}/10000 | L1: {l1} | VRAM: {vram} MB", flush=True)
                except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
                    pass
            time.sleep(check_interval)
        print(f"• ACT-B training process {pid} has concluded.", flush=True)
    else:
        print("• No active ACT-B training process found. Checking for existing checkpoint...", flush=True)

    if not best_checkpoint.is_dir():
        raise RuntimeError(f"ACT-B best_offline checkpoint directory not found at {best_checkpoint}")
    print(f"  ✓ ACT-B best_offline checkpoint verified at: {best_checkpoint}", flush=True)


def train_act_g(args: argparse.Namespace):
    log_phase("Stage 2/3: Training Goal-Conditioned ACT Policy (ACT-G)")
    output_dir = Path(args.act_g_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_offline"

    if best_checkpoint.is_dir() and not args.force_retrain:
        print(f"• ACT-G best_offline checkpoint already exists at {best_checkpoint}. Skipping training.", flush=True)
        return

    python_bin = sys.executable
    cmd = [
        python_bin,
        "scripts/train_policy.py",
        "--policy-type", "act-g",
        "--dataset-dir", str(args.train_dataset),
        "--val-dataset-dir", str(args.val_dataset),
        "--output-dir", str(output_dir),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--eval-freq", str(args.eval_freq),
        "--save-freq", str(args.save_freq),
        "--seed", str(args.seed),
    ]

    log_path = Path("outputs") / "train_act_g.log"
    ret = run_command_stream(cmd, log_file=log_path)
    if ret != 0:
        raise RuntimeError(f"ACT-G training failed with exit code {ret}. Check {log_path} for details.")

    if not best_checkpoint.is_dir():
        raise RuntimeError(f"ACT-G training completed but best_offline checkpoint was not found at {best_checkpoint}")
    print(f"  ✓ ACT-G best_offline checkpoint verified at: {best_checkpoint}", flush=True)


def evaluate_policies(args: argparse.Namespace):
    log_phase("Stage 3/3: Closed-Loop MuJoCo Rollout Evaluation")
    python_bin = sys.executable

    eval_results = {}
    policies = [
        ("act-b", Path(args.act_b_dir) / "best_offline", Path("outputs/eval_rollouts/act_b_nominal_v1")),
        ("act-g", Path(args.act_g_dir) / "best_offline", Path("outputs/eval_rollouts/act_g_nominal_v1")),
    ]

    for p_name, p_path, out_dir in policies:
        print(f"\n--- Benchmarking {p_name.upper()} ({args.eval_episodes} episodes, seeds {args.eval_seed_start}..{args.eval_seed_start + args.eval_episodes - 1}) ---", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            python_bin,
            "scripts/eval_policy.py",
            "--policy-path", str(p_path),
            "--episodes", str(args.eval_episodes),
            "--seed-start", str(args.eval_seed_start),
            "--goal-provider", "auto",
            "--render-video",
            "--output-dir", str(out_dir),
        ]
        log_path = out_dir / "eval_execution.log"
        ret = run_command_stream(cmd, log_file=log_path)
        if ret != 0:
            raise RuntimeError(f"Evaluation of {p_name} failed with exit code {ret}.")

        summary_file = out_dir / "eval_summary.json"
        if not summary_file.is_file():
            raise RuntimeError(f"Expected evaluation summary at {summary_file} not found.")
        with open(summary_file) as f:
            eval_results[p_name] = json.load(f)

    # Produce Consolidated Benchmark Comparison
    comparison_dir = Path("outputs/eval_rollouts")
    comparison_file = comparison_dir / "benchmark_comparison.json"
    comparison_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "evaluation_config": {
            "episodes": args.eval_episodes,
            "seed_start": args.eval_seed_start,
            "held_out_seeds": list(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes)),
        },
        "policies": eval_results,
    }
    with open(comparison_file, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\n✓ Consolidated benchmark comparison saved to: {comparison_file}", flush=True)

    # Write completion flag
    flag_file = Path("outputs/PIPELINE_COMPLETE.json")
    flag_data = {
        "status": "COMPLETED",
        "completed_at": datetime.now(UTC).isoformat(),
        "summary": {
            "act_b_success_rate": eval_results["act-b"]["success_rate_pct"],
            "act_b_wilson_ci": eval_results["act-b"]["wilson_ci_95"],
            "act_g_success_rate": eval_results["act-g"]["success_rate_pct"],
            "act_g_wilson_ci": eval_results["act-g"]["wilson_ci_95"],
        },
        "artifacts": {
            "act_b_checkpoint": str(Path(args.act_b_dir) / "best_offline"),
            "act_g_checkpoint": str(Path(args.act_g_dir) / "best_offline"),
            "comparison_report": str(comparison_file),
        },
    }
    with open(flag_file, "w") as f:
        json.dump(flag_data, f, indent=2)
    print(f"✓ Pipeline completion manifest written to: {flag_file}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous pipeline runner for ACT-B and ACT-G training & evaluation.")
    parser.add_argument("--act-b-dir", type=str, default="outputs/checkpoints/act_b_nominal_v1")
    parser.add_argument("--act-g-dir", type=str, default="outputs/checkpoints/act_g_nominal_v1")
    parser.add_argument("--train-dataset", type=str, default="data/nominal_train_v1")
    parser.add_argument("--val-dataset", type=str, default="data/nominal_val_v1")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-start", type=int, default=2000)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    start_time = time.time()

    log_phase("AUTONOMOUS PRODUCTION PIPELINE: ACT-B & ACT-G")
    print(f"• Start Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"• Target Updates per Policy: {args.steps:,}")
    print(f"• Evaluation: {args.eval_episodes} episodes on held-out seeds {args.eval_seed_start}..{args.eval_seed_start + args.eval_episodes - 1}")

    # Stage 1: Wait for ACT-B
    wait_for_act_b(Path(args.act_b_dir))

    # Stage 2: Train ACT-G
    train_act_g(args)

    # Stage 3: Closed-Loop MuJoCo Evaluation for both
    evaluate_policies(args)

    elapsed_min = (time.time() - start_time) / 60.0
    log_phase(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_min:.1f} MINUTES")


if __name__ == "__main__":
    main()
