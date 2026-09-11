# Project Memory: LeRobot Reliability Lab

> Living project memory tracking hardware topologies, environment specifications, verified milestones, and operational constraints.

---

## 1. System Identity & Topology

- **Project**: LeRobot Reliability Lab (`lerobot-reliability`)
- **Package Path**: `src/lerobot_reliability/`
- **Current Phase**: Phase 0 (Operational Foundation & Scaffolding)
- **Primary Technical Report**: `docs/LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md` (Frozen v1.0)
- **Research Landscape Snapshot**: `docs/FRONTIER_RESEARCH_SUPPORT.md` (Frozen v1.0)

### Dual-Compute Architecture

| Attribute | Local Development PC | Remote Compute Worker (`workstation`) |
|---|---|---|
| **Role** | Code authoring, CI, unit testing, git management | Full VLA inference, containerized benchmarks, evaluations |
| **GPU** | NVIDIA GeForce RTX 3070 (8 GB VRAM) | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **Driver / CUDA** | Driver 595.84 / CUDA 13.2 | Driver 550.163.01 / CUDA 12.4 |
| **Python** | 3.12.3 (in `.venv`) | not provisioned (old venv removed 2026-09-10) |
| **PyTorch** | 2.10.0+cu128 | TBD (rebuild via `uv` on first `--push`) |
| **LeRobot** | 0.6.1 | TBD |
| **MuJoCo** | 3.12.0 | TBD |
| **Bridge Tool** | `scripts/sync_worker.py` | SSH host `workstation` → `umcai@100.75.252.120` |

---

## 2. Remote Workstation Protocol

- **SSH Alias**: `workstation` (`~/.ssh/config` → `umcai@100.75.252.120`, key `~/.ssh/id_ed25519`). Host: `umcai-workstation`.
- **Remote Repo Directory**: `/home/umcai/lerobot-reliability` (does not exist yet; `sync_worker.py --push` creates it). The pre-pivot `/home/umcai/medical_ai_projects/lerobot-agentic` tree was deleted 2026-09-10.
- **Git Security Invariant**: Git credentials remain strictly on the local PC. Code pushes bake the current local commit SHA into `.git_commit` before transfer so remote runs retain full audit provenance.
- **Sync Commands**:
  - `python scripts/sync_worker.py --status`: Checks GPU telemetry and active Python jobs.
  - `python scripts/sync_worker.py --push`: Synchronizes code, tests, and configs.
  - `python scripts/sync_worker.py --exec "<cmd>"`: Runs commands on the remote GPU.
  - `python scripts/sync_worker.py --pull`: Pulls run bundles, checkpoints, and videos.

---

## 3. Verified Milestones

- **2026-09-08**: v1.0 Technical Report and Frontier Research Support documents authored, red-teamed, and frozen (per their own audit stamp).
- **2026-09-09**: Clean-slate reset committed (`23d9d5f`) — old `lerobot_agentic` codebase and docs removed, frozen docs added.
- **2026-09-09**: Operational foundation scaffolded:
  - Package renamed to `lerobot-reliability` in `pyproject.toml`; installed editable via `uv pip install -e ".[dev,lerobot]"`.
  - `scripts/sync_worker.py` restored; SSH to `workstation` (RTX 4090) confirmed.
  - Root docs drafted (`README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `MEMORY.md`); safe `.env.example` created; `.gitignore` protections verified.
- **2026-09-10**: Phase 0 consolidation:
  - Consistency sweep: single project name (`LeRobot Reliability Lab` / `lerobot-reliability`); `requirements.txt` removed (`pyproject.toml` + `uv.lock` are the only source of truth); `setuptools>=77` build floor.
  - Stale docs resolved: `docs/DL-pipeline/` deleted; `docs/algorithm/` de-staled and retained as background reference; `04_spatial_grounding_voxposer.md` deleted (no role in the frozen architecture).
  - `scripts/download_assets.py` lint-clean; still downloads nothing by default (asset-wiring is the next mini phase).
  - Remote pre-pivot folder `/home/umcai/medical_ai_projects/lerobot-agentic` (11 GB, old ACT checkpoints + eval videos) deleted; `sync_worker.py` repointed to `/home/umcai/lerobot-reliability`.
  - `ruff check .` clean; `pytest tests/` green.
- **2026-09-10**: Phase 1 — canonical data models & interface contracts (branch `feat/phase-1-contracts`):
  - Pre-flight: `vla-eval` 0.5.0 installed & imports; ships `benchmarks/{libero,libero_plus,libero_pro,robocasa,robocasa365,robomme,vlabench,...}`, `runners/{sync,live}_runner.py`, `model_servers/`, `cli/_docker.py` — matches the frozen architecture's E0 assumption.
  - Pre-flight: all 9 `download_assets.py` HF repo IDs resolve. **License follow-ups**: `lerobot/pi05_libero_base` is under the **Gemma** license (not Apache); `lerobot/libero_plus` has **no declared license** on its card — confirm before Phase 5.
  - `src/lerobot_reliability/data_types.py` + `protocols/` package + `tests/test_contracts.py`; 48 tests green, `ruff` clean.
  - Decision: canonical arrays are `numpy.ndarray` (not framework tensors) — neutral across runtime/telemetry/analysis, matches `vla-eval` wire + Parquet.
