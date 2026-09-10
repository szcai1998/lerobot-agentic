"""tests/test_environment.py.

Verifies the operational foundation layer, package setup, dependency imports,
hardware detection, and secrets isolation contracts.
"""

import os
from pathlib import Path

import lerobot
import mujoco
import pyarrow
import pydantic
import torch

import lerobot_reliability

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_package_import():
    """Verify lerobot_reliability imports cleanly with correct version."""
    assert lerobot_reliability.__version__ == "0.1.0"


def test_core_dependencies():
    """Verify required robotics and ML libraries are installed with expected versions."""
    assert lerobot.__version__ == "0.6.1"
    assert mujoco.__version__.startswith("3.12")
    assert int(pydantic.__version__.split(".")[0]) >= 2
    assert int(pyarrow.__version__.split(".")[0]) >= 15


def test_torch_cuda():
    """Verify PyTorch detects CUDA and reports valid device properties if available."""
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU: {device_name} ({vram_gb:.2f} GB VRAM)")
        assert vram_gb > 0
        # Basic tensor allocation
        x = torch.ones((10, 10), device="cuda")
        assert x.sum().item() == 100.0


def test_secrets_isolation():
    """Verify .env and transient files are strictly covered in .gitignore (Rule 5)."""
    gitignore_path = REPO_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist"
    gitignore_content = gitignore_path.read_text()
    
    # Must ignore .env
    assert ".env" in gitignore_content, ".env must be ignored by .gitignore"
    # Must ignore outputs and media
    assert "outputs/" in gitignore_content
    assert "*.parquet" in gitignore_content


def test_sync_worker_exists_and_executable():
    """Verify synchronization tooling exists and has execution permissions."""
    sync_script = REPO_ROOT / "scripts" / "sync_worker.py"
    assert sync_script.exists(), "scripts/sync_worker.py must exist"
    assert os.access(sync_script, os.X_OK), "scripts/sync_worker.py must be executable"


def test_single_source_of_dependency_truth():
    """pyproject.toml + uv.lock are the only dependency manifests (no requirements.txt)."""
    assert not (REPO_ROOT / "requirements.txt").exists(), (
        "requirements.txt must not exist; use pyproject.toml + uv.lock"
    )
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert (REPO_ROOT / "uv.lock").exists()


def test_no_legacy_package_name_in_source():
    """The pre-pivot package name `lerobot_agentic` must not reappear in src/ or scripts/."""
    for base in ("src", "scripts"):
        for path in (REPO_ROOT / base).rglob("*.py"):
            assert "lerobot_agentic" not in path.read_text(), f"legacy name in {path}"
