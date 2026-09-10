#!/usr/bin/env python3
"""scripts/download_assets.py.

Pre-flight asset inventory and caching management utility for LeRobot Reliability Lab.
Allows listing, sizing, inspecting, and selectively downloading policy checkpoints
and benchmark datasets defined in the Technical Report and Research Support documents.

DOES NOT download anything unless explicitly invoked with --download <key>.
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
        description="Pretrained ACT policy on Aloha simulation transfer cube task",
    ),
    # 2. Modern Modifiable VLA (SmolVLA suite)
    "smolvla_libero": AssetSpec(
        key="smolvla_libero",
        repo_id="lerobot/smolvla_libero",
        repo_type="model",
        role="Primary VLA (Standard LIBERO)",
        est_size_mb=906.0,
        description="SmolVLA weights and normalizers evaluated on LIBERO suite",
    ),
    "smolvla_libero_plus": AssetSpec(
        key="smolvla_libero_plus",
        repo_id="lerobot/smolvla_libero_plus",
        repo_type="model",
        role="Primary VLA (Robustness)",
        est_size_mb=906.0,
        description="SmolVLA weights for perturbation analysis on LIBERO-plus",
    ),
    "smolvla_robocasa": AssetSpec(
        key="smolvla_robocasa",
        repo_id="lerobot/smolvla_robocasa",
        repo_type="model",
        role="Primary VLA (Non-LIBERO Confirmation)",
        est_size_mb=906.0,
        description="SmolVLA weights evaluated on RoboCasa kitchen manipulation",
    ),
    # 3. High-Capacity Frozen Reference VLA (RTX 4090 evaluation)
    "pi05_libero": AssetSpec(
        key="pi05_libero",
        repo_id="lerobot/pi05_libero_base",
        repo_type="model",
        role="Frozen Reference VLA (Cross-Policy Target)",
        est_size_mb=14467.0,
        description="π0.5 high-capacity flow-matching policy evaluated on LIBERO",
    ),
    # 4. Benchmark Datasets & Environments
    "libero_spatial": AssetSpec(
        key="libero_spatial",
        repo_id="lerobot/libero_spatial_image",
        repo_type="dataset",
        role="Spatial Benchmark Dataset",
        est_size_mb=6296.0,
        description="LIBERO Spatial manipulation demonstrations and sensory data",
    ),
    "libero_10": AssetSpec(
        key="libero_10",
        repo_id="lerobot/libero_10_image",
        repo_type="dataset",
        role="Long-Horizon Benchmark Dataset",
        est_size_mb=5120.0,
        description="LIBERO 10 long-horizon demonstration dataset",
    ),
    "libero_plus": AssetSpec(
        key="libero_plus",
        repo_id="lerobot/libero_plus",
        repo_type="dataset",
        role="Perturbation Benchmark Dataset",
        est_size_mb=4600.0,
        description="LIBERO-plus visual and physical perturbation benchmarks",
    ),
    "robocasa_human": AssetSpec(
        key="robocasa_human",
        repo_id="lerobot/robocasa_target_human_unified",
        repo_type="dataset",
        role="Kitchen Benchmark Dataset",
        est_size_mb=15360.0,
        description="RoboCasa human demonstrations for kitchen manipulation tasks",
    ),
}


def is_asset_cached(spec: AssetSpec, cache_dir: str | None = None) -> bool:
    """Checks if the repository files already exist in local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        filename = "config.json" if spec.repo_type == "model" else ".gitattributes"
        res = try_to_load_from_cache(
            repo_id=spec.repo_id,
            filename=filename,
            repo_type=spec.repo_type,
            cache_dir=cache_dir,
        )
        return isinstance(res, str)
    except Exception:  # noqa: BLE001 - best-effort cache probe, never fatal
        return False


def list_assets(cache_dir: str | None = None):
    """Prints a formatted inventory table of all assets with their status."""
    print("=" * 105)
    print(f"{'Key':<20} {'Repo ID':<42} {'Type':<8} {'Size (MB)':<12} {'Status':<10} {'Role'}")
    print("=" * 105)

    total_size = 0.0
    for key, spec in ASSET_INVENTORY.items():
        cached = is_asset_cached(spec, cache_dir)
        status_str = "CACHED" if cached else "PENDING"
        print(f"{key:<20} {spec.repo_id:<42} {spec.repo_type:<8} {spec.est_size_mb:<12.1f} {status_str:<10} {spec.role}")
        total_size += spec.est_size_mb

    print("-" * 105)
    print(f"Total Inventory Size: {total_size / 1024:.2f} GB across {len(ASSET_INVENTORY)} assets.")
    print("=" * 105)


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


def download_asset(key: str, cache_dir: str | None = None):
    """Downloads a single asset from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    if key not in ASSET_INVENTORY:
        print(f"❌ Unknown asset key '{key}'. Available keys: {list(ASSET_INVENTORY.keys())}")
        sys.exit(1)

    spec = ASSET_INVENTORY[key]
    print("=" * 80)
    print(f"📥 Starting download for [{key}]: {spec.repo_id} ({spec.est_size_mb:.1f} MB)...")
    print("=" * 80)

    try:
        path = snapshot_download(
            repo_id=spec.repo_id,
            repo_type=spec.repo_type,
            cache_dir=cache_dir,
        )
        print(f"✅ Successfully cached {spec.repo_id} to: {path}")
    except Exception as e:  # noqa: BLE001 - surface any hub/network/disk failure and abort
        print(f"❌ Failed to download {spec.repo_id}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight asset inventory and caching utility (DOES NOT download by default)."
    )
    parser.add_argument("--list", action="store_true", help="List all assets and cache status.")
    parser.add_argument("--dry-run", action="store_true", help="Perform pre-flight dry-run check.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom HF cache directory.")
    parser.add_argument("--download", type=str, default=None, help="Explicit asset key to download.")

    args = parser.parse_args()

    if args.download:
        download_asset(args.download, cache_dir=args.cache_dir)
    elif args.list:
        list_assets(cache_dir=args.cache_dir)
    else:
        # Default behavior is safe dry-run
        dry_run(cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
