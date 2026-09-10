# Implementation Roadmap: LeRobot Reliability Lab

> Phased engineering roadmap for building the runtime reliability and recovery laboratory.

Reference: [`docs/LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md`](docs/LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md).

---

## MVP cut line

The **smallest publishable version** of this project is Phases 0–4 plus the
SmolVLA slice of Phase 5: one modifiable VLA, one LIBERO-family benchmark through
the preferred backend, the heuristic + one learned detector, `reset_reinfer` +
`truncate_correct` + one semantic recovery expert, achievement verification in the
loop, and reproducible run bundles — all on the single RTX 4090. Everything beyond
that (the strong frozen VLA, RoboCasa confirmation, the extended recovery family,
the cross-policy study) is **incremental evidence, not a release blocker**. Scope
creep past this line during Phases 1–4 should be pushed to a later phase.

---

## Phase 0: Operational Foundation & Environment Scaffolding *(Current)*

- [x] Package reconfiguration (`pyproject.toml`) and editable installation with `uv`.
- [x] Environment alignment on Python 3.12, PyTorch 2.10, LeRobot 0.6.1, MuJoCo 3.12.0.
- [x] Dual-compute synchronization tooling (`scripts/sync_worker.py`) connecting local RTX 3070 to remote RTX 4090 worker.
- [x] Living documentation initialization (`README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `MEMORY.md`, `AGENTS.md`).
- [x] Baseline environment and security sanity tests (`tests/test_environment.py`).
- [x] Phase 0 consolidation: name/dependency/doc consistency sweep, stale-doc removal, lint-clean scripts, remote cleanup.

### Phase 0.5: Asset-download wiring *(next mini phase — no downloads)*

- [ ] Finalize `scripts/download_assets.py` inventory against the frozen policy/benchmark roster (repo IDs, sizes, roles, licences).
- [ ] `--dry-run` / `--list` reporting reviewed; `--download <key>` path tested against one small asset only when explicitly greenlit.
- [ ] **No bulk dataset or checkpoint downloads in this phase.**

---

## Phase 1: Canonical Data Models & Interface Contracts

**Goal**: Establish pure, typed contracts with 0% circular dependencies and 100% test coverage.

- [ ] **Data Models** (`src/lerobot_reliability/data_types.py`):
  - `CanonicalObservation` (named camera views, depth, proprioception, instruction).
  - `CanonicalActionView` & `ExecutableAction` (strict non-invertibility design).
  - `ExecutionHistory` (bounded rolling buffer).
  - `FailureEstimate`, `RecoveryProposal`, `RecoveryCost`, `AchievementResult`.
- [ ] **Protocols** (`src/lerobot_reliability/protocols/`):
  - `PolicyAdapter` & `PolicyCapabilities`.
  - `RuntimeBenchmarkAdapter` vs. `EvaluationOracle` (hard privileged separation).
  - `FailureDetector`, `RecoveryExpert`, `AchievementVerifier`, `ExecutionMemory`.
- [ ] **Contract Tests** (`tests/test_contracts.py`):
  - Verify type safety, serialization, and protocol conformances.

---

## Phase 2: Fail-Closed Checkpoint & Configuration Integrity

**Goal**: Eliminate silent failure modes from misconfigured weights or normalization.

- [ ] **Manifest Schemas** (`src/lerobot_reliability/integrity/manifests.py`):
  - `PolicyManifest`, `BenchmarkManifest`, `EvaluationBackendManifest`, `ComputeManifest`.
- [ ] **Checksum & Integrity Engine** (`src/lerobot_reliability/integrity/checker.py`):
  - SHA-256 computation on weights, preprocessors, and normalization stats.
  - Fail-closed validation logic aborting on mismatch or fallback.
- [ ] **Deterministic Known-Output Smoke Test Harness**:
  - Validates forward pass consistency on fixed reference fixtures.

---

## Phase 3: Telemetry, Event Logging & Run-Bundle Exporter

**Goal**: Standardize event timing, raw cost accounting, and reproducible exports.

- [ ] **Monotonic Event Logging** (`src/lerobot_reliability/telemetry/events.py`):
  - Nanosecond precision $TTD$ and $TTR$ calculation.
  - Streaming JSONL writers for `failure_events.jsonl` and `recovery_events.jsonl`.
- [ ] **Cost Accounting** (`src/lerobot_reliability/telemetry/cost.py`):
  - Wall-clock ms, GPU ms, queue disruptions, and billable token/API tracking.
- [ ] **Parquet Telemetry Sink** (`src/lerobot_reliability/telemetry/parquet_sink.py`):
  - High-frequency sensory and intervention logging.
- [ ] **Run-Bundle Exporter** (`src/lerobot_reliability/telemetry/exporter.py`):
  - Generates immutable run directories matching Appendix A of the Technical Report.

---

## Phase 4: Minimal Sandbox & Baseline Recovery Loop

**Goal**: Complete the end-to-end execution loop in a fast, deterministic environment.

- [ ] **Lightweight Sandbox** (`src/lerobot_reliability/sim/sandbox.py`):
  - Fast, deterministic simulated test environment for CI and unit tests.
- [ ] **Baseline Mechanism Library**:
  - Heuristic failure detector (contact and aperture boundaries).
  - `reset_reinfer` expert (queue flush and fresh inference).
  - `truncate_correct` expert (chunk invalidation and horizon adjustment).
  - `AchievementVerifier` with `"achieved"`, `"not_achieved"`, and `"uncertain"` support.
- [ ] **Closed-Loop Integration Test** (`tests/test_recovery_loop.py`):
  - Step $\rightarrow$ Detect $\rightarrow$ Truncate $\rightarrow$ Intervene $\rightarrow$ Verify $\rightarrow$ Export Bundle.

---

## Phase 5: Policy & Upstream Benchmark Integrations

**Goal**: Connect real policies and community manipulation benchmarks.

- [ ] **Policy Adapters** (`src/lerobot_reliability/adapters/policies/`):
  - ACT adapter (local sanity, fast execution on RTX 3070).
  - SmolVLA adapter (mandatory modern modifiable VLA).
  - MolmoAct2 / $\pi_{0.5}$ adapter (frozen high-capacity reference for RTX 4090).
- [ ] **Evaluation Backends** (`src/lerobot_reliability/backends/`):
  - `VlaEvalBackend`: AllenAI `vla-evaluation-harness` containerized integration for standard LIBERO, LIBERO-plus, and RoboCasa365.
  - `DirectRuntimeBackend`: Native LeRobot bridge for fine-grained queue/state interventions.
- [ ] **Benchmark Calibration & Reproduction**:
  - Validate nominal policy success against upstream baselines.

---

## Phase 6: Comparative Recovery Study & Public Deliverables

**Goal**: Generate empirical findings and release public artifacts.

- [ ] **Evaluation Matrix Execution**:
  - Run comparative experiments across recovery families under matched seeds.
- [ ] **HUD Video Renderer** (`scripts/render_rollout.py`):
  - Overlay telemetry (detector score, active recovery expert, verification state).
- [ ] **Public Deliverables**:
  - Clean GitHub release with 1-line quickstarts and documentation.
  - Technical Report / preprint drafting.
