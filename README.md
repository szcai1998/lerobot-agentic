# LeRobot Reliability Lab

> A Reproducible Reliability and Recovery Framework for Modern Robot Policies on a Single RTX 4090.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![LeRobot 0.6.1](https://img.shields.io/badge/lerobot-0.6.1-orange.svg)](https://github.com/huggingface/lerobot)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 1. Overview

Modern vision-language-action (VLA) and imitation learning (IL) policies achieve high nominal performance on closed-world benchmarks, but remain brittle during runtime execution: target objects shift mid-chunk, grasps slip, camera observations delay, and policies enter irrecoverable states while continuing to execute obsolete action queues.

**LeRobot Reliability Lab** is an open-source systems and research-engineering framework built on the Hugging Face [LeRobot](https://github.com/huggingface/lerobot) ecosystem. It provides:

1. **Shared Policy Integration Contracts**: Common interfaces across heterogeneous policies (ACT, SmolVLA, MolmoAct2, $\pi_{0.5}$).
2. **Upstream Evaluation Substrate**: Reuses the AllenAI [`vla-evaluation-harness`](https://github.com/allenai/vla-evaluation-harness) (`vla-eval`) for containerized execution across LIBERO, LIBERO-plus, and RoboCasa365, with a direct runtime fallback for fine-grained action-chunk invalidation.
3. **Standardized Telemetry & Event Schemas**: Nanosecond-precision monotonic timing for Time-to-Detect ($TTD$) and Time-to-Recover ($TTR$), raw API usage tracking, and Parquet telemetry streams.
4. **Composable Recovery Interventions**: A unified `RecoveryExpert` library (reset/re-infer, detect/truncate/correct, retry/backoff, semantic recovery) evaluated under matched seeds and perturbations.
5. **Achievement Verification**: Runtime outcome verification distinguishing commanded actions from physical outcomes ($\text{commanded} \neq \text{achieved}$), treating `"uncertain"` as a first-class result.
6. **Fail-Closed Checkpoint Integrity**: Strict cryptographic verification of model weights, preprocessors, and normalization statistics before any evaluation begins.
7. **Single-GPU Reproducibility**: All mandatory benchmarks and pipelines strictly respect a $1 \times \text{NVIDIA RTX 4090 (24 GB VRAM)}$ envelope.

For full architectural and landscape details, consult:
- [`docs/LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md`](docs/LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md): Frozen v1.0 technical specification.
- [`docs/FRONTIER_RESEARCH_SUPPORT.md`](docs/FRONTIER_RESEARCH_SUPPORT.md): Frozen v1.0 landscape snapshot, evidence classes, and prior art tracking.

---

## 2. Hardware Architecture & Dual-Compute Setup

This project uses a dual-compute workflow:
- **Local Development PC**: NVIDIA GeForce RTX 3070 (8 GB VRAM) — used for development, linting, unit testing, and lightweight sandbox tests.
- **Remote Workstation Worker**: NVIDIA GeForce RTX 4090 (24 GB VRAM) via SSH (`workstation`) — used for full VLA policy execution, containerized benchmarks, and heavy rollouts.

Tooling in `scripts/sync_worker.py` bridges the local Git-controlled codebase and the remote GPU compute worker automatically.

---

## 3. Quickstart

### Prerequisites
- Python 3.12+
- `uv` (recommended) or `pip`
- CUDA 12.4+ supported GPU

### Installation

```bash
# Clone the repository (the GitHub slug `lerobot-agentic` is historical; the
# project and Python package are named `lerobot-reliability`)
git clone https://github.com/szcai1998/lerobot-agentic.git
cd lerobot-agentic

# Create and activate virtual environment using uv
uv venv .venv --python 3.12
source .venv/bin/activate

# Install package in editable mode with LeRobot and dev dependencies
uv pip install -e ".[dev,lerobot]"
```

### Environment Configuration

Copy the sample environment variables:
```bash
cp .env.example .env
```
Populate `.env` with required API keys (e.g. `GEMINI_API_KEY` for semantic recovery).

### Running Tests & Linting

```bash
# Run linters
ruff check .

# Run test suite
pytest tests/ -v
```

---

## 4. Remote GPU Worker Synchronization

To sync code to and from the RTX 4090 workstation:

```bash
# Check remote worker GPU status and active jobs
python scripts/sync_worker.py --status

# Push local codebase and configs to remote worker (bakes git commit for provenance)
python scripts/sync_worker.py --push

# Execute a command on the remote worker
python scripts/sync_worker.py --exec "pytest tests/ -v"

# Pull evaluation outputs, run bundles, and videos from remote worker
python scripts/sync_worker.py --pull
```

---

## 5. Repository Structure

```text
lerobot-reliability/
├── README.md                                   # Project overview and quickstart
├── ARCHITECTURE.md                             # High-level architecture summary
├── ROADMAP.md                                  # Implementation phases
├── AGENTS.md                                   # Operational constitution & Karpathy rules
├── MEMORY.md                                   # Persistent project state & hardware specs
├── pyproject.toml                              # Build and package definition
├── docs/
│   ├── LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md  # Frozen v1.0 technical report
│   ├── FRONTIER_RESEARCH_SUPPORT.md                 # Frozen v1.0 research landscape
│   └── ...
├── src/lerobot_reliability/                     # Core framework package
│   ├── data_types.py                           # (Phase 1) canonical frozen value types
│   ├── protocols/                              # (Phase 1) runtime-checkable interface contracts
│   ├── backends/                               # (Phase 5) VlaEvalBackend & DirectRuntimeBackend
│   ├── adapters/                               # (Phase 5) PolicyAdapter and reliability bridges
│   ├── detectors/                              # (Phase 4) FailureDetector implementations
│   ├── recovery/                               # (Phase 4) RecoveryManager & RecoveryExpert library
│   ├── integrity/                              # (Phase 2) Checkpoint and processor verification
│   ├── telemetry/                              # (Phase 3) Parquet metrics and event logging
│   └── analysis/                               # (Phase 3) Statistics and run-bundle loaders
├── configs/                                    # Experiment, policy, and benchmark configs
├── scripts/                                    # sync_worker.py, verification, and runners
├── tests/                                      # Unit, contract, and integration tests
└── outputs/                                    # Run bundles, telemetry, videos (gitignored)
```

---

## 6. Scientific Position & Scope Boundaries

As formalized in `docs/FRONTIER_RESEARCH_SUPPORT.md`, LeRobot Reliability Lab adheres to a **Guilty Until Proven** standard:
- Generic VLA orchestration is occupied prior art (RoboBRIDGE, EmbodiedSkills).
- Event-triggered replanning and action-chunk resets are occupied prior art (VLA-Corrector, FLARE).
- Deployment-facing reliability protocols are occupied prior art (ROEP).
- Our contribution is the **reliability lifecycle and reproducible evidence contract**: fail-closed model/processor integrity, runtime-vs-oracle interface separation, standardized failure/recovery telemetry, explicit achievement verification, cost accounting, and reproducible run bundles across heterogeneous policies on an accessible prosumer GPU.
