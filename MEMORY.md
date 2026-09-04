# MEMORY.md: Project Memory & Operational State

This file records the current project status, active hardware profile, verified technical invariants, and ongoing roadmaps for `lerobot-agentic`.

---

## 1. Project Profile & Active Metadata

- **Repository:** `szcai1998/lerobot-agentic`
- **Local Path:** `/home/aivise/Documents/antigravity/lerobot-agentic`
- **Domain:** Embodied AI, Hierarchical Vision-Language-Action (VLA), Imitation Learning, Physics Simulation
- **Last Verified Date:** September 2026
- **Current Development Phase:** Stage 1 (Environment Setup & Harness Foundation)

---

## 2. Hardware Profile & Compute Topology

| Compute Node | Device Specification | Role in Architecture |
| :--- | :--- | :--- |
| **Local Edge Node** | **NVIDIA GeForce RTX 3070 (8 GB VRAM)** | • DeepMind MuJoCo EGL headless rendering (>400 FPS)<br/>• PyTorch ACTPolicy local training (~2.4 GB VRAM, 50k steps in ~45 min)<br/>• Real-time policy inference at 50 Hz (~1.4 GB VRAM) |
| **Cloud Cognitive Tier** | **Google Gemini Robotics ER (`gemini-robotics-er-2-preview`)** | • Multi-modal scene perception & 1–2 Hz task decomposition<br/>• Normalized 2D spatial bounding box regression `[0, 1000]`<br/>• Visual anomaly detection & closed-loop recovery |
| **Local Vision Fallback** | **OpenCV Color & Contour Affordance Tracker** | • Zero-cloud offline fallback emitting identical Pydantic schemas |

---

## 3. Verified System Invariants

1. **Python Environment:**
   - Python 3.11.15 in `.venv/` managed with `uv`.
   - PyTorch `2.14.0+cu130` with CUDA GPU acceleration verified (`torch.cuda.is_available() == True`).
   - Hugging Face `lerobot` compatible with Python 3.11 (`lerobot>=0.4.0,<0.6.0`).
2. **Gemini Developer API:**
   - API key stored safely in `.env` (gitignored).
   - Live authentication verified on `gemini-robotics-er-2-preview` and `gemini-2.5-flash`.
   - Real-time multimodal grounding verified on MuJoCo RGB camera frames.
3. **DeepMind MuJoCo Physics Sim:**
   - MuJoCo 3.12.0 with EGL headless GPU rendering (`MUJOCO_GL=egl`).
   - 6-DoF arm model with parallel gripper (`embodied_arm.xml`).
   - Joint addresses safely mapped via `model.jnt_qposadr`.
4. **Test Suite:**
   - Unit tests passing 100% (`pytest tests/`).

---

## 4. Architectural Documents & Living Artifacts

- [`ARCHITECTURE.md`](./ARCHITECTURE.md): Master high-level evergreen architecture map.
- [`ROADMAP.md`](./ROADMAP.md): Bite-sized 5-stage task roadmap with inputs, outputs, constraints, and success criteria.
- [`AGENTS.md`](./AGENTS.md): Operational constitution (Karpathy Rules, verification ladder, physics safety, one-line pointers).
- [`docs/subsystem/`](./docs/subsystem/): Comprehensive subsystem design documents.
- [`docs/algorithm/`](./docs/algorithm/): Academic literature mathematical derivations (ACT, Diffusion, SmolVLA, VoxPoser).
- [`docs/DL-pipeline/`](./docs/DL-pipeline/): Deep learning training specifications, dataset formulations, and hyperparameters.

---

## 5. Current Task & Next Actionable Steps

1. **Finish Stage 1 Harness Verification:**
   - Complete `uv pip install -e ".[lerobot,dev]"` and verify `import lerobot`.
   - Update `AGENTS.md`, `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`.
2. **Proceed to Stage 2 (Dual-Camera Sim & Dynamic Arena):**
   - Add eye-in-hand `wrist_cam` directly to gripper palm.
   - Add tabletop object randomization and receptacle target zone.
