#!/usr/bin/env python3
"""
scripts/sync_worker.py
Automates synchronization between the local Git-controlled repository and the
remote high-performance GPU compute worker (workstation RTX 4090).

Maintains strict separation:
- Git & GitHub: 100% on the local machine (no credentials needed on remote).
- Compute & VRAM: 100% on the remote workstation (native NVMe SSD and RTX 4090).
"""

import argparse
import subprocess
import sys
from pathlib import Path

LOCAL_REPO = Path(__file__).resolve().parent.parent
REMOTE_HOST = "workstation"
REMOTE_DIR = "/home/umcai/medical_ai_projects/lerobot-agentic"


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"🚀 Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def push_to_remote():
    """Syncs code, scripts, tests, and datasets from local PC to remote worker."""
    print("=" * 80)
    print("📤 Pushing local codebase and datasets to remote worker...")
    print("=" * 80)

    # 1. Bake local git commit SHA into .git_commit for remote provenance
    commit_file = LOCAL_REPO / ".git_commit"
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO, text=True
        ).strip()
        commit_file.write_text(commit_sha + "\n")
        print(f"  ✓ Baked Git commit {commit_sha[:10]} into .git_commit")
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  ⚠️ Could not determine git commit: {e}")

    # 2. Rsync code and data, strictly excluding .git and .venv
    rsync_cmd = [
        "rsync",
        "-avz",
        "--delete",
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=outputs",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.pytest_cache",
        "--exclude=.ruff_cache",
        f"{LOCAL_REPO}/",
        f"{REMOTE_HOST}:{REMOTE_DIR}/",
    ]
    res = run_cmd(rsync_cmd)
    if res.returncode == 0:
        print(f"✅ Successfully synced local repository to {REMOTE_HOST}:{REMOTE_DIR}")


def pull_from_remote(checkpoint_only: bool = False):
    """Pulls checkpoints, telemetry, and rollout videos from remote worker to local repo."""
    print("=" * 80)
    print("📥 Pulling trained outputs and checkpoints from remote worker...")
    print("=" * 80)

    remote_src = f"{REMOTE_HOST}:{REMOTE_DIR}/outputs/"
    local_dst = LOCAL_REPO / "outputs/"
    local_dst.mkdir(parents=True, exist_ok=True)

    if checkpoint_only:
        remote_src = f"{REMOTE_HOST}:{REMOTE_DIR}/outputs/checkpoints/"
        local_dst = LOCAL_REPO / "outputs/checkpoints/"

    rsync_cmd = [
        "rsync",
        "-avzP",
        remote_src,
        str(local_dst) + "/",
    ]
    res = run_cmd(rsync_cmd)
    if res.returncode == 0:
        print(f"✅ Successfully downloaded outputs to {local_dst}")


def check_remote_status():
    """Queries GPU status and running training processes on the remote worker."""
    print("=" * 80)
    print("🖥️  Remote Worker Status (NVIDIA GeForce RTX 4090):")
    print("=" * 80)
    remote_cmd = (
        "echo '=== GPU Telemetry ===' && "
        "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu "
        "--format=csv,noheader && "
        "echo '=== Active Policy Training Jobs ===' && "
        "(pgrep -af 'scripts/train_policy.py' || echo 'No active training job found.')"
    )
    subprocess.run(["ssh", REMOTE_HOST, remote_cmd], check=False)


def main():
    parser = argparse.ArgumentParser(description="Synchronize code and checkpoints with remote GPU worker.")
    parser.add_argument("--push", action="store_true", help="Push local code/data to remote worker.")
    parser.add_argument("--pull", action="store_true", help="Pull trained checkpoints/outputs from remote worker.")
    parser.add_argument("--status", action="store_true", help="Check remote GPU and process status.")
    parser.add_argument(
        "--checkpoints-only", action="store_true", help="When pulling, only sync outputs/checkpoints/."
    )

    args = parser.parse_args()

    if not (args.push or args.pull or args.status):
        parser.print_help()
        sys.exit(1)

    if args.push:
        push_to_remote()
    if args.pull:
        pull_from_remote(checkpoint_only=args.checkpoints_only)
    if args.status:
        check_remote_status()


if __name__ == "__main__":
    main()
