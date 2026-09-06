# MEMORY.md: Project Memory & Operational State

This file records the current project status, active hardware profile, verified technical invariants, and ongoing roadmaps for `lerobot-agentic`.

---

## 1. Project Profile & Active Metadata

- **Repository:** `szcai1998/lerobot-agentic`
- **Local Path:** `/home/aivise/Documents/antigravity/lerobot-agentic`
- **Domain:** Embodied AI, Hierarchical Vision-Language-Action (VLA), Imitation Learning, Physics Simulation
- **Last Verified Date:** September 2026
- **Current Development Phase:** Stage 2 (Dual-Camera Sim & Cognitive Perception) — Stage 1 Verified Complete ✅

---

## 2. Hardware Profile & Compute Topology

| Compute Node | Device Specification | Role in Architecture |
| :--- | :--- | :--- |
| **Local Edge Node** | **NVIDIA GeForce RTX 3070 (8 GB VRAM)** | • DeepMind MuJoCo EGL headless rendering (>400 FPS)<br/>• PyTorch ACTPolicy local training (~3.5–5.5 GB VRAM planning envelope; validate in Gate 0)<br/>• Real-time policy inference at 50 Hz (~1.2–1.4 GB VRAM) |
| **Cloud Cognitive Tier** | **Google Gemini Robotics ER (`gemini-robotics-er-2-preview`)** | • Multi-modal scene perception & 1–2 Hz task decomposition<br/>• Normalized 2D spatial bounding box regression `[0, 1000]`<br/>• Visual anomaly detection & closed-loop recovery |
| **Local Vision Fallback** | **OpenCV Color & Contour Affordance Tracker** | • Zero-cloud offline fallback emitting identical Pydantic schemas |

---

## 3. Verified System Invariants

1. **Python Environment & Reproducibility:**
   - Python 3.11.15 in `.venv/` managed with `uv`.
   - Deterministic lockfile `uv.lock` generated and committed (targeting LeRobot 0.6.4, PyTorch 2.6.0, MuJoCo 3.12.0).
   - PyTorch `2.14.0+cu130` with CUDA GPU acceleration verified (`torch.cuda.is_available() == True`).
2. **Gemini Developer API:**
   - API key stored safely in `.env` (gitignored).
   - Standardized strictly on `gemini-robotics-er-2-preview` with exponential backoff retries. Dead fallback models (`gemini-2.0-flash`, `gemini-2.5-flash`) pruned.
   - Real-time multimodal grounding verified on MuJoCo RGB camera frames.
3. **DeepMind MuJoCo Physics Sim & Model Hierarchy:**
   - MuJoCo 3.12.0 with EGL headless GPU rendering (`MUJOCO_GL=egl`).
   - 6-DoF arm model with parallel gripper (`embodied_arm.xml`) tracked under version control.
   - Eye-in-hand `<camera name="wrist_cam">` and `<site name="ee_site">` attached directly to `<body name="gripper_base">`.
   - Dual-camera rendering (`overhead_cam` + `wrist_cam`) verified in `MuJoCoRobotEnv`.
   - Joint addresses safely mapped via `model.jnt_qposadr`.
4. **Visuomotor Policy & Stock LeRobot ACT Integration:**
   - Goal conditioning vector $\mathbf{g}_t \in \mathbb{R}^{11}$ mapped to stock LeRobot ACT's native `observation.environment_state` (`FeatureType.ENV`) projected by `encoder_env_state_input_proj` (zero custom library fork).
   - `policy.reset()` queue flush edge-triggered via `replan_id` / `replan_consumed` tracking in `AtomicPlanState`.
   - Actuator-specific command limits: arm joints 1..6 clipped to $\pm 3.14159$, linear gripper slide joint clipped to $[-0.025, 0.025]\,\text{m}$.
   - 7-stage Pick-and-Place FSM (`PREGRASP` -> `APPROACH` -> `GRASP` -> `LIFT` -> `TRANSPORT` -> `PLACE` -> `RETREAT`) implemented for Oracle and System A.
5. **Thread Safety & Clock Pacing:**
   - Multi-threaded rendering race prevented via `LatestFrameBuffer`: only the simulation thread accesses MuJoCo `MjData` and `Renderer`, pushing immutable frame copies to the supervisor.
   - 50 Hz real-time wall-clock pacing enforced for cloud-in-the-loop Systems C & D (`time.perf_counter()`), ensuring 5.0 simulated seconds equal 5.0 physical seconds.
6. **Test Suite & Code Hygiene:**
   - Unit tests passing 100% (`pytest tests/ -v`: 18/18 passed).
   - Linting clean (`ruff check .`: 0 errors).
   - Simulation rollout passing (`run_rollout.py --steps 50`).

---

## 4. Architectural Documents & Living Artifacts

- [`ARCHITECTURE.md`](./ARCHITECTURE.md): Master high-level evergreen architecture map.
- [`ROADMAP.md`](./ROADMAP.md): Bite-sized 5-stage task roadmap with inputs, outputs, constraints, and success criteria.
- [`AGENTS.md`](./AGENTS.md): Operational constitution (Karpathy Rules, verification ladder, physics safety, one-line pointers).
- [`docs/dossier_01_lerobot_agentic.md`](./docs/dossier_01_lerobot_agentic.md): Fully audited student-ready dossier with Section 6 Architectural Reference Scaffold.
- [`docs/subsystem/`](./docs/subsystem/): Comprehensive subsystem design documents.
- [`docs/algorithm/`](./docs/algorithm/): Academic literature mathematical derivations (ACT, Diffusion, SmolVLA, VoxPoser).
- [`docs/DL-pipeline/`](./docs/DL-pipeline/): Deep learning training specifications, dataset formulations, and hyperparameters.

---

## 5. Current Task & Next Actionable Steps

1. **All 16 Audit Items & P0-P2 Defects Resolved:**
   - Fixed PyTorch compatibility contract (`torch>=2.7.0,<2.12.0`, `torchvision>=0.22.0,<0.27.0`) and updated SDK to `google-genai>=2.0.0`.
   - Corrected LeRobot import paths to `from lerobot.policies.act import ACTConfig, ACTPolicy`.
   - Mapped goal conditioning vector to native stock ACT `observation.environment_state` (`FeatureType.ENV`).
   - Isolated MuJoCo rendering to simulation thread via `LatestFrameBuffer`.
   - Enforced 50 Hz real-time loop pacing for Systems C and D.
   - Implemented 7-stage Pick-and-Place FSM for Oracle and System A.
   - Enforced per-actuator limits (arm $\pm 3.14$, gripper $\pm 0.025\,\text{m}$).
   - Edge-triggered recovery resets (`replan_id`) to eliminate repetitive policy flushes.
   - Fixed destination receptacle position ($[0.32, -0.15, 0.43]\,\text{m}$) and strengthened Pydantic `Literal` schema.
   - Renamed disturbance to "deterministic mid-trajectory state displacement perturbation".
2. **Next Steps (Gate 0 Execution by Student):**
   - Student executes Gate 0 checklist in a clean subshell with `uv sync`.
   - Validate RTX 3070 VRAM planning envelope during ACT forward passes and EGL dual-camera rendering.

