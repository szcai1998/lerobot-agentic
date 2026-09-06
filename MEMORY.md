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
4. **Visuomotor Policy & Recovery Queue Flush:**
   - `policy.reset()` queue flush implemented in `VisuomotorPolicyExecutor` to prevent stale chunk execution during replanning.
5. **Test Suite:**
   - Unit tests passing 100% (`pytest tests/`: 5/5 passed).
   - Simulation rollout passing (`run_rollout.py --steps 50`).

---

## 4. Architectural Documents & Living Artifacts

- [`ARCHITECTURE.md`](./ARCHITECTURE.md): Master high-level evergreen architecture map.
- [`ROADMAP.md`](./ROADMAP.md): Bite-sized 5-stage task roadmap with inputs, outputs, constraints, and success criteria.
- [`AGENTS.md`](./AGENTS.md): Operational constitution (Karpathy Rules, verification ladder, physics safety, one-line pointers).
- [`docs/dossier_01_lerobot_agentic.md`](./docs/dossier_01_lerobot_agentic.md): Fully audited 10/10 student-ready dossier with Section 6 Architectural Reference Scaffold.
- [`docs/subsystem/`](./docs/subsystem/): Comprehensive subsystem design documents.
- [`docs/algorithm/`](./docs/algorithm/): Academic literature mathematical derivations (ACT, Diffusion, SmolVLA, VoxPoser).
- [`docs/DL-pipeline/`](./docs/DL-pipeline/): Deep learning training specifications, dataset formulations, and hyperparameters.

---

## 5. Current Task & Next Actionable Steps

1. **Strategic Reframing & Dossier 01 Complete (10/10 Student-Ready):**
   - Completed all 16 audit feedback items and P0 requirements across all sections of `docs/dossier_01_lerobot_agentic.md`.
   - Built complete Architectural Reference Scaffold in Section 6 (valid 6-DoF arm MJCF with wrist camera/actuators/ee_site, async supervisor thread with `AtomicPlanState`, dynamic camera unprojection with `INVALID_DEPTH` handling, LeRobot 0.6+ `PolicyProcessorPipeline`, `policy.reset()` queue flush, Oracle/A/B/C/D dispatch, Wilson 95% CIs, scenario manifest logging, Gate 0 verification checklist).
   - Synchronized core codebase: `embodied_arm.xml`, `sim/env.py`, `cognitive/supervisor.py`, `cognitive/schemas.py`, `policy/executor.py`, and `uv.lock`.
2. **Next Steps (Gate 0 & Stage 2 in ROADMAP.md):**
   - Verify Gate 0 checklist execution in clean subshell.
   - Dual-camera simulation arena setup (`overhead_cam` + `wrist_cam`) with domain randomization.
   - Grounding integration and offline CV fallback.

