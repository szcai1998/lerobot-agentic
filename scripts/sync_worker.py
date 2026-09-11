#!/usr/bin/env python3
"""scripts/sync_worker.py.

Automates synchronization and remote execution between the local Git-controlled
repository and the high-performance remote GPU compute worker (NVIDIA RTX 4090).

Maintains strict separation:
- Git, GitHub & Local Dev: 100% on the local machine (no credentials needed on remote).
- Compute & Large VLA Execution: 100% on the remote workstation (24 GB RTX 4090).
"""

import argparse
import subprocess
import sys
from pathlib import Path

LOCAL_REPO = Path(__file__).resolve().parent.parent
DEFAULT_REMOTE_HOST = "workstation"
DEFAULT_REMOTE_DIR = "/home/umcai/lerobot-agentic"


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"🚀 Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def bake_git_commit() -> str:
    """Bakes local Git HEAD commit SHA into .git_commit for remote run provenance."""
    commit_file = LOCAL_REPO / ".git_commit"
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO, text=True
        ).strip()
        commit_file.write_text(commit_sha + "\n")
        print(f"  ✓ Baked Git commit {commit_sha[:10]} into .git_commit")
        return commit_sha
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  ⚠️ Could not determine git commit: {e}")
        return "unknown"


def push_to_remote(remote_host: str, remote_dir: str):
    """Syncs code, scripts, tests, configs, and assets from local PC to remote worker."""
    print("=" * 80)
    print(f"📤 Pushing local codebase to remote worker ({remote_host}:{remote_dir})...")
    print("=" * 80)

    # 1. Record provenance
    bake_git_commit()

    # 2. Rsync code and configs, excluding .git, .venv, outputs, and caches
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
        f"{remote_host}:{remote_dir}/",
    ]
    res = run_cmd(rsync_cmd)
    if res.returncode == 0:
        print(f"✅ Successfully synced local repository to {remote_host}:{remote_dir}")


def pull_from_remote(remote_host: str, remote_dir: str, checkpoint_only: bool = False):
    """Pulls evaluation run bundles, checkpoints, and videos from remote worker."""
    print("=" * 80)
    print(f"📥 Pulling outputs from remote worker ({remote_host}:{remote_dir})...")
    print("=" * 80)

    remote_src = f"{remote_host}:{remote_dir}/outputs/"
    local_dst = LOCAL_REPO / "outputs/"
    local_dst.mkdir(parents=True, exist_ok=True)

    if checkpoint_only:
        remote_src = f"{remote_host}:{remote_dir}/outputs/checkpoints/"
        local_dst = LOCAL_REPO / "outputs/checkpoints/"
        local_dst.mkdir(parents=True, exist_ok=True)

    rsync_cmd = [
        "rsync",
        "-avzP",
        remote_src,
        str(local_dst) + "/",
    ]
    res = run_cmd(rsync_cmd)
    if res.returncode == 0:
        print(f"✅ Successfully downloaded outputs to {local_dst}")


def check_remote_status(remote_host: str):
    """Queries GPU status and running training/evaluation processes on the remote worker."""
    print("=" * 80)
    print(f"🖥️  Remote Worker Status ({remote_host}):")
    print("=" * 80)
    remote_cmd = (
        "echo '=== GPU Telemetry ===' && "
        "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu "
        "--format=csv,noheader && "
        "echo '=== Active Python Processes ===' && "
        "(pgrep -af 'python' | grep -v 'gnome' | grep -v 'system' || echo 'No active python jobs found.')"
    )
    subprocess.run(["ssh", remote_host, remote_cmd], check=False)


def execute_remote(remote_host: str, remote_dir: str, command: str):
    """Executes a command on the remote worker inside the repository with .venv activated."""
    print("=" * 80)
    print(f"⚡ Executing on {remote_host}:{remote_dir}: {command}")
    print("=" * 80)
    remote_cmd = (
        f"cd {remote_dir} && "
        f"source .venv/bin/activate 2>/dev/null || true && "
        f"{command}"
    )
    subprocess.run(["ssh", "-t", remote_host, remote_cmd], check=False)


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize code and execute workloads on remote GPU worker."
    )
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST, help="SSH host alias (default: workstation)")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="Remote project directory")
    parser.add_argument("--push", action="store_true", help="Push local code/configs to remote worker.")
    parser.add_argument("--pull", action="store_true", help="Pull outputs/checkpoints from remote worker.")
    parser.add_argument("--status", action="store_true", help="Check remote GPU and process status.")
    parser.add_argument(
        "--checkpoints-only", action="store_true", help="When pulling, only sync outputs/checkpoints/."
    )
    parser.add_argument("--exec", dest="exec_cmd", type=str, help="Execute command on remote worker.")

    args = parser.parse_args()

    if not (args.push or args.pull or args.status or args.exec_cmd):
        parser.print_help()
        sys.exit(1)

    if args.push:
        push_to_remote(args.remote_host, args.remote_dir)
    if args.pull:
        pull_from_remote(args.remote_host, args.remote_dir, checkpoint_only=args.checkpoints_only)
    if args.status:
        check_remote_status(args.remote_host)
    if args.exec_cmd:
        execute_remote(args.remote_host, args.remote_dir, args.exec_cmd)


if __name__ == "__main__":
    main()
