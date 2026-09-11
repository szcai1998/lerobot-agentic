#!/usr/bin/env python3
"""scripts/download_assets.py.

Pre-flight asset inventory and caching management utility for LeRobot Reliability Lab.
Allows listing, sizing, inspecting, and selectively downloading policy checkpoints
and benchmark datasets defined in the Technical Report and Research Support documents.

DOES NOT download anything unless explicitly invoked with --download <key> or
--download-all / --download-all-datasets / --download-all-models.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure .env is loaded for HF_TOKEN (minimal stdlib parser: KEY=VALUE, '#' comments,
# tolerant of '=' inside the value and surrounding single/double quotes).
env_file = REPO_ROOT / ".env"
if env_file.exists():
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass
class AssetSpec:
    key: str
    repo_id: str
    repo_type: str  # "model" or "dataset"
    role: str
    est_size_mb: float
    license: str
    description: str


# Canonical Asset Inventory from Technical Report & Frontier Research Support
ASSET_INVENTORY: dict[str, AssetSpec] = {
    # 1. Historical Baseline Policy
    "act_aloha": AssetSpec(
        key="act_aloha",
        repo_id="lerobot/act_aloha_sim_transfer_cube_human",
        repo_type="model",
        role="Baseline Sanity (Local Dev / CI)",
        est_size_mb=206.0,
        license="apache-2.0",
        description="Pretrained ACT policy on Aloha simulation transfer cube task",
    ),
    # 2. Modern Modifiable VLA (SmolVLA suite)
    "smolvla_libero": AssetSpec(
        key="smolvla_libero",
        repo_id="lerobot/smolvla_libero",
        repo_type="model",
        role="Primary VLA (Standard LIBERO)",
        est_size_mb=906.0,
        license="apache-2.0",
        description="SmolVLA weights and normalizers evaluated on LIBERO suite",
    ),
    "smolvla_libero_plus": AssetSpec(
        key="smolvla_libero_plus",
        repo_id="lerobot/smolvla_libero_plus",
        repo_type="model",
        role="Primary VLA (Robustness)",
        est_size_mb=906.0,
        license="apache-2.0",
        description="SmolVLA weights for perturbation analysis on LIBERO-plus",
    ),
    "smolvla_robocasa": AssetSpec(
        key="smolvla_robocasa",
        repo_id="lerobot/smolvla_robocasa",
        repo_type="model",
        role="Primary VLA (Non-LIBERO Confirmation)",
        est_size_mb=906.0,
        license="apache-2.0",
        description="SmolVLA weights evaluated on RoboCasa kitchen manipulation",
    ),
    # 3. High-Capacity Frozen Reference VLA (RTX 4090 evaluation)
    "pi05_libero": AssetSpec(
        key="pi05_libero",
        repo_id="lerobot/pi05_libero_base",
        repo_type="model",
        role="Frozen Reference VLA (Cross-Policy Target)",
        est_size_mb=14467.0,
        license="gemma",
        description="π0.5 high-capacity flow-matching policy evaluated on LIBERO",
    ),
    # 4. Benchmark Datasets & Environments
    "libero_spatial": AssetSpec(
        key="libero_spatial",
        repo_id="lerobot/libero_spatial_image",
        repo_type="dataset",
        role="Spatial Benchmark Demonstrations",
        est_size_mb=6296.0,
        license="apache-2.0",
        description="LIBERO Spatial manipulation demonstrations and sensory data",
    ),
    "libero_object": AssetSpec(
        key="libero_object",
        repo_id="lerobot/libero_object_image",
        repo_type="dataset",
        role="Object Benchmark Demonstrations",
        est_size_mb=8821.4,
        license="apache-2.0",
        description="LIBERO Object variation manipulation demonstrations",
    ),
    "libero_goal": AssetSpec(
        key="libero_goal",
        repo_id="lerobot/libero_goal_image",
        repo_type="dataset",
        role="Goal Conditioning Demonstrations",
        est_size_mb=6013.1,
        license="apache-2.0",
        description="LIBERO Goal conditioning manipulation demonstrations",
    ),
    "libero_10": AssetSpec(
        key="libero_10",
        repo_id="lerobot/libero_10_image",
        repo_type="dataset",
        role="Long-Horizon Benchmark Demonstrations",
        est_size_mb=5120.0,
        license="apache-2.0",
        description="LIBERO 10 long-horizon demonstration dataset",
    ),
    "libero_plus": AssetSpec(
        key="libero_plus",
        repo_id="lerobot/libero_plus",
        repo_type="dataset",
        role="Perturbation Benchmark Dataset",
        est_size_mb=4600.0,
        license="unknown",
        description="LIBERO-plus visual and physical perturbation benchmarks",
    ),
    "robocasa_human": AssetSpec(
        key="robocasa_human",
        repo_id="lerobot/robocasa_target_human_unified",
        repo_type="dataset",
        role="Kitchen Benchmark Human Demonstrations",
        est_size_mb=15360.0,
        license="apache-2.0",
        description="RoboCasa human demonstrations for kitchen manipulation tasks",
    ),
    "aloha_sim_transfer_cube": AssetSpec(
        key="aloha_sim_transfer_cube",
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        repo_type="dataset",
        role="Baseline Sanity Human Demonstrations",
        est_size_mb=66.6,
        license="mit",
        description="Aloha transfer cube human demonstrations for ACT baseline training",
    ),
    "aloha_sim_insertion": AssetSpec(
        key="aloha_sim_insertion",
        repo_id="lerobot/aloha_sim_insertion_human",
        repo_type="dataset",
        role="Contact-Rich Baseline Demonstrations",
        est_size_mb=87.1,
        license="mit",
        description="Aloha peg insertion human demonstrations for ACT baseline training",
    ),
}


def get_cached_repos(cache_dir: str | None = None) -> set[tuple[str, str]]:
    """Returns set of (repo_type, repo_id) tuples present in local HF cache."""
    try:
        from huggingface_hub import scan_cache_dir

        hf_cache = scan_cache_dir(cache_dir=cache_dir)
        return {(repo.repo_type, repo.repo_id) for repo in hf_cache.repos}
    except Exception:  # noqa: BLE001 - cache probe fallback
        return set()


def list_assets(cache_dir: str | None = None):
    """Prints a formatted inventory table of all assets with their status."""
    cached_set = get_cached_repos(cache_dir)

    print("=" * 125)
    print(
        f"{'Key':<24} {'Repo ID':<42} {'Type':<8} {'Size (MB)':<11} {'License':<12} {'Status':<10} {'Role'}"
    )
    print("=" * 125)

    total_size = 0.0
    for key, spec in ASSET_INVENTORY.items():
        cached = (spec.repo_type, spec.repo_id) in cached_set
        status_str = "CACHED" if cached else "PENDING"
        print(
            f"{key:<24} {spec.repo_id:<42} {spec.repo_type:<8} {spec.est_size_mb:<11.1f} {spec.license:<12} {status_str:<10} {spec.role}"
        )
        total_size += spec.est_size_mb

    print("-" * 125)
    print(
        f"Total Inventory Size: {total_size / 1024:.2f} GB across {len(ASSET_INVENTORY)} assets "
        f"({sum(1 for s in ASSET_INVENTORY.values() if s.repo_type == 'model')} models, "
        f"{sum(1 for s in ASSET_INVENTORY.values() if s.repo_type == 'dataset')} datasets)."
    )
    print("=" * 125)


def check_auth():
    """Verifies Hugging Face authentication token status."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()
        print(f"✅ Hugging Face Hub Authenticated as: '{user.get('name')}' (type: {user.get('type')})")
        return True
    except Exception as e:  # noqa: BLE001 - diagnostic only; any failure means "not authed"
        print(f"⚠️ Hugging Face Hub: Unauthenticated or invalid token ({e})")
        return False


def dry_run(cache_dir: str | None = None):
    """Performs dry-run feasibility check without downloading any data."""
    print("\n🔍 Running Pre-Flight Asset Feasibility Check (DRY RUN)...")
    check_auth()
    list_assets(cache_dir)
    print("\n💡 DRY RUN COMPLETE: Zero bytes were downloaded.")
    print("To download specific assets when ready, run:")
    print("  python scripts/download_assets.py --download <key>")
    print("  python scripts/download_assets.py --download-all-datasets")


def download_asset(key: str, cache_dir: str | None = None) -> bool:
    """Downloads a single asset from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    if key not in ASSET_INVENTORY:
        print(f"❌ Unknown asset key '{key}'. Available keys: {list(ASSET_INVENTORY.keys())}")
        return False

    spec = ASSET_INVENTORY[key]
    print("=" * 80)
    print(f"📥 Starting download for [{key}]: {spec.repo_id} ({spec.est_size_mb:.1f} MB, {spec.license})...")
    print("=" * 80)

    try:
        path = snapshot_download(
            repo_id=spec.repo_id,
            repo_type=spec.repo_type,
            cache_dir=cache_dir,
        )
        print(f"✅ Successfully cached {spec.repo_id} to: {path}")
        return True
    except Exception as e:  # noqa: BLE001 - surface any hub/network/disk failure
        print(f"❌ Failed to download {spec.repo_id}: {e}")
        return False


def download_batch(repo_type_filter: str | None = None, cache_dir: str | None = None):
    """Downloads all assets matching the filter."""
    targets = [
        (k, s)
        for k, s in ASSET_INVENTORY.items()
        if repo_type_filter is None or s.repo_type == repo_type_filter
    ]
    total_mb = sum(s.est_size_mb for _, s in targets)
    label = repo_type_filter or "all"
    print("=" * 80)
    print(f"📦 Batch download requested for [{label}] assets: {len(targets)} assets (~{total_mb / 1024:.2f} GB)")
    print("=" * 80)

    failed = []
    for idx, (k, s) in enumerate(targets, 1):
        print(f"\n[{idx}/{len(targets)}] Downloading {k} ({s.repo_id})...")
        success = download_asset(k, cache_dir=cache_dir)
        if not success:
            failed.append(k)

    print("\n" + "=" * 80)
    if failed:
        print(f"⚠️ Batch download completed with {len(failed)} failures: {failed}")
        sys.exit(1)
    else:
        print(f"✅ Batch download complete! All {len(targets)} assets cached successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight asset inventory and caching utility (DOES NOT download by default)."
    )
    parser.add_argument("--list", action="store_true", help="List all assets and cache status.")
    parser.add_argument("--dry-run", action="store_true", help="Perform pre-flight dry-run check.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom HF cache directory.")
    parser.add_argument("--download", type=str, default=None, help="Explicit asset key to download.")
    parser.add_argument(
        "--download-all-datasets", action="store_true", help="Download all benchmark demonstration datasets."
    )
    parser.add_argument(
        "--download-all-models", action="store_true", help="Download all policy checkpoint models."
    )
    parser.add_argument("--download-all", action="store_true", help="Download all inventory assets.")

    args = parser.parse_args()

    if args.download:
        success = download_asset(args.download, cache_dir=args.cache_dir)
        if not success:
            sys.exit(1)
    elif args.download_all_datasets:
        download_batch(repo_type_filter="dataset", cache_dir=args.cache_dir)
    elif args.download_all_models:
        download_batch(repo_type_filter="model", cache_dir=args.cache_dir)
    elif args.download_all:
        download_batch(repo_type_filter=None, cache_dir=args.cache_dir)
    elif args.list:
        list_assets(cache_dir=args.cache_dir)
    else:
        # Default behavior is safe dry-run
        dry_run(cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
