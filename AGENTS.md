# AGENTS.md: Development & Operational Constitution for `lerobot-agentic`

> Universal operational constitution, behavioral constraints, robotics engineering rules, and verification standards for autonomous AI coding agents in `lerobot-agentic`.

---

## 🧭 Living Document Directory (One-Line Pointers)

Before performing any action or proposing changes, consult these living repositories of truth:
- **System Architecture**: [`ARCHITECTURE.md`](./ARCHITECTURE.md) — Evergreen high-level map, dual-rate topology, and module interfaces.
- **Project Roadmap**: [`ROADMAP.md`](./ROADMAP.md) — Discrete 5-stage agile implementation roadmap with strict success criteria and scope boundaries.
- **Operational Memory**: [`MEMORY.md`](./MEMORY.md) — Real-time project status, active compute profile (RTX 3070 8GB), and verified technical invariants.
- **Subsystem Deep Dives**: [`docs/subsystem/`](./docs/subsystem/) — Architectural specifications for Cognitive (01), Policy (02), Simulation (03), and Dataset/Eval (04) tiers.
- **Algorithm Formulations**: [`docs/algorithm/`](./docs/algorithm/) — Academic literature and mathematical compendium (ACT, Diffusion, SmolVLA, VoxPoser).
- **Deep Learning Pipeline**: [`docs/DL-pipeline/`](./docs/DL-pipeline/) — CVAE Transformer architecture, episodic tensor schemas, and RTX 3070 training guide.

---

## 1. Core Behavioral Controls (The Karpathy Rules)

Coding agents suffer from a "defaults problem": a systemic bias to barrel ahead, hallucinate requirements, over-engineer abstractions, and make unrequested renovations. You MUST strictly obey these five core principles:

### Rule 1: Think Before Coding (Never Barrel Ahead)
* **Surface Assumptions Explicitly**: Before touching code, state your technical assumptions and architectural design.
* **Halt on Ambiguity**: When requirements are ambiguous, contradictory, or underspecified, **STOP and ask the user**. Never silently guess intent.
* **Present Trade-offs**: When multiple viable approaches exist, present the options and trade-offs rather than unilaterally choosing.
* **Expose Confusion**: If existing code appears buggy, inconsistent, or confusing, state your confusion explicitly. Do not invent fragile workarounds to paper over latent confusion.

### Rule 2: Simplicity First (Write 50 Lines, Not 500)
* **Minimum Viable Code**: Implement strictly what was requested. No speculative abstractions, premature design patterns, or unrequested generic wrappers.
* **YAGNI (You Aren't Gonna Need It)**: The hallmark of a senior engineer is simplicity. The simplest code that solves the problem and passes verification is the best code.
* **Prevent Context Degradation**: Lean, focused files protect the AI context window and maintain long-term repository health.

### Rule 3: Surgical Edits & Living Documentation
* **Strict Minimal Blast Radius**: Touch ONLY lines strictly necessary to satisfy the request. Every edit must be directly traceable to the user's instruction.
* **Zero Unsolicited Renovations**: NEVER reformat untouched lines, reorganize folder structures, clean up whitespace in adjacent functions, or refactor unrelated code.
* **Synchronous Documentation Sync**: Whenever you add, modify, or deprecate public interfaces, API routes, CLI flags, or environment variables, you MUST update the corresponding documentation (`README.md`, `ARCHITECTURE.md`, `MEMORY.md`, `.env.example`) in the exact same PR. Prevent doc drift.
* **Preserve Documentation**: Retain all existing comments and docstrings. Clean up only the temporary logs, variables, or imports that *your* changes introduced or orphaned.

### Rule 4: Reproduction-First, Anti-Flailing & Invariant Tests
* **Reproduction First**: Before writing a bugfix, write a minimal reproduction test proving the failure exists. Confirm it fails for the expected reason.
* **Diagnose Before Patching**: Trace the stack and identify root cause before modifying code. "Shotgun debugging" (guessing and checking) is strictly forbidden.
* **The Invariant Test Rule**: NEVER weaken, comment out, or alter existing test assertions to make a build pass. If code fails a test, the code is wrong.
* **3-Strike Anti-Flailing Circuit Breaker**: If a fix fails after **2 consecutive attempts**, STOP immediately. Re-evaluate your assumptions, explain the obstacle to the developer, and request human guidance rather than looping blindly. If an approach fails completely, run `git reset --hard HEAD` to return to a clean state.

### Rule 5: Zero-Trust Security & Secrets Isolation
* **Zero Credential Exposure**: NEVER print, cat, echo, log, or commit secret keys (`GEMINI_API_KEY`), passwords, private certificates, or `.env` files into terminal output, commits, or PR descriptions.
* **Pre-Staging Gitignore Check**: Before staging, verify that all sensitive and transient files (`.env`, `*.mp4`, checkpoints, test dumps) are strictly covered in `.gitignore`.
* **Safe Configuration Placeholders**: Add new configuration keys to `.env.example` with safe dummy placeholders; never store real operational secrets in version control.

---

## 2. Embodied AI & Robotics Invariants

In addition to universal engineering rules, all code within `lerobot-agentic` must strictly adhere to these physics and robotics laws:

### 2.1 Kinematics & Physics Safety
* **Safe Joint Addressing**: NEVER assume robot joint indices map directly to `qpos[:7]`. Freejoints (e.g., floating manipulands or receptacles) introduce dynamic indexing offsets. Always resolve joint indices using:
  ```python
  model.jnt_qposadr[model.joint(joint_name).id]
  ```
* **EGL Headless Support**: Always enforce `os.environ.setdefault("MUJOCO_GL", "egl")` before importing OpenGL or initializing MuJoCo contexts to prevent X11 display crashes on headless servers.
* **Action Clamping**: All target joint positions emitted by motor policies must be clipped to actuator joint limits specified in `embodied_arm.xml`.

### 2.2 Dual-Rate Control Hierarchy
* **Frequency Decoupling**: Maintain strict separation between control rates:
  - **Cognitive Supervisory Tier**: 1–2 Hz (Cloud Gemini Robotics ER or local CV fallback). Never run LLM/VLM calls inside high-frequency motor loops.
  - **Visuomotor Motor Policy Tier**: 50 Hz ($\Delta t = 20\,	ext{ms}$, chunk size $K=50$, 1.0s horizon).
  - **Physics Simulation Tier**: 500 Hz internal integration ($10 	imes 0.002\,	ext{s}$ substeps per control step).
* **Strong Typing**: All cognitive plan exchanges must strictly conform to Pydantic models in `lerobot_agentic.cognitive.schemas.SpatialGroundingPlan`.

---

## 3. Package Manager & Dependency Discipline

* **Deterministic Dependency Tooling**: Use `uv` exclusively for package management in `.venv`:
  - Install core and dev dependencies: `uv pip install -e ".[lerobot,dev]"`
  - Package listings: `uv pip list`
* **Python 3.11 Compatibility**: Ensure all Hugging Face `lerobot` dependencies remain compatible with the active Python 3.11 environment (`lerobot>=0.4.0,<0.6.0`).
* **Zero Global Pollution**: Never run global `pip install` or `sudo apt` commands without explicit user instruction.

---

## 4. Verification Ladder & Definition of Done (DoD)

Before declaring any task or stage complete, verify every rung of this ladder:

1. **Rung 0: Static Analysis**: Run syntax check and linters (`ruff check .`).
2. **Rung 1: Security & Secrets**: Verify `.env` is uncommitted and no raw keys are printed.
3. **Rung 2: Unit Test Suite**: Execute `.venv/bin/pytest tests/` (must pass 100%).
4. **Rung 3: Simulation Rollout**: Execute `python scripts/run_rollout.py --steps 50` to guarantee zero physics divergence or segmentation faults.
5. **Rung 4: Documentation Synchronization**: Ensure `README.md`, `ARCHITECTURE.md`, and `MEMORY.md` reflect the exact current state.
