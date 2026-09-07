# MEMORY.md: Project Memory & Operational State

This file records the current project status, active hardware profile, verified technical invariants, and ongoing roadmaps for `lerobot-agentic`.

---

## 1. Project Profile & Active Metadata

- **Repository:** `szcai1998/lerobot-agentic`
- **Local Path:** `/home/aivise/Documents/antigravity/lerobot-agentic`
- **Domain:** Embodied AI, Hierarchical Vision-Language-Action (VLA), Imitation Learning, Physics Simulation
- **Last Verified Date:** September 2026
- **Current Development Phase:** Phase 2 Part 2 Active ⏳ — Dual ACT Training Infrastructure & Invariant Test Suite Complete & Smoke-Tested ✅ (ACT-B & ACT-G) — Proceeding to 10k Production Runs

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
   - Python 3.12.3 in `.venv/` managed with `uv`.
   - Deterministic lockfile `uv.lock` targeting `lerobot==0.6.1`, `torch>=2.7.0,<2.12.0`, `mujoco==3.12.0`, `av==15.1.0`, `datasets==4.8.5`.
   - PyTorch with CUDA GPU acceleration verified (`torch.cuda.is_available() == True`, RTX 3070 8GB).
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
   - 13-DoF goal conditioning vector $\mathbf{g}_t \in \mathbb{R}^{13}$ (target xyz, dest xyz, 7-class one-hot across canonical primitives `["reach", "grasp", "lift", "transport", "place", "retreat", "recover"]`) mapped to stock LeRobot ACT's native `observation.environment_state` (`FeatureType.ENV`) projected by `encoder_env_state_input_proj` (zero custom library fork).
   - Non-privileged policy conditioning: Target position $\hat{\mathbf{p}}_{\text{target}}$ is derived strictly via overhead RGB-D camera unprojection (`CameraGeometry`) at phase transitions, ensuring zero privileged state leakage into policy inputs.
   - Decoupled expert execution: MuJoCo GT is strictly isolated to 6D IK expert trajectory execution and offline validation error auditing ($\|\mathbf{p}_{\text{obs}} - \mathbf{p}_{\text{GT}}\|$).
   - LeRobot 0.6+ pre/post processor pipeline restored via `make_pre_post_processors(policy_cfg=..., pretrained_path=...)`; environment boundary emits unbatched tensors `(3, H, W)`, `(7,)`, `(13,)` so LeRobot preprocessor owns batching.
   - Primary benchmark mode: Queue / Receding Horizon (`chunk_size=50`, `n_action_steps=10`, `temporal_ensemble_coeff=None`), operating at 50 Hz control, ~5 Hz inference cadence.
   - `policy.reset()` queue flush edge-triggered via `replan_id` / `replan_consumed` tracking in `AtomicPlanState`.
   - Actuator-specific command limits: arm joints 1..6 clipped to $\pm 3.14159$, linear gripper slide joint clipped to $[-0.025, 0.025]\,\text{m}$.
   - 6D Pose IK with downward orientation constraint ($R \in SO(3)$) and secondary nullspace posture projection implemented in `ClassicalIKController` for Oracle and System A.
   - 7-stage Pick-and-Place FSM (`PREGRASP` -> `APPROACH` -> `GRASP` -> `LIFT` -> `TRANSPORT` -> `PLACE` -> `RETREAT`).
5. **Thread Safety, Clock Pacing & Termination:**
   - Multi-threaded rendering race prevented via `LatestFrameBuffer`: only the simulation thread accesses MuJoCo `MjData` and `Renderer`, pushing immutable frame copies to the supervisor.
   - 50 Hz real-time wall-clock pacing enforced for cloud-in-the-loop Systems C & D (`time.perf_counter()`), ensuring simulated time tracks physical wall-clock time.
   - Standardized 15.0 s (750 steps @ 50 Hz) episode timeout with multi-condition early termination (`task_complete`, `unrecoverable_failure`, `timeout`, `software_halt`).
   - `SpatialGroundingPlan` uses `decision_note`; advisory software halt explicitly separated from hardware safety E-stop.
   - Standardized on LeRobotDataset v3.0 with explicit `dataset.finalize()` lifecycle call.
6. **Demonstration Dataset Verification & Sanity Audits (Phase 2 Part 1):**
   - `data/nominal_train_v1`: 50 accepted episodes, 16,578 frames @ 50 Hz.
     - SHA-256: `fab14fe3e6f49b6dc8eb0af653b52f0ee07c9beef36851ef0b75771a50aef67f`.
     - Physical thresholds: Receptacle distance mean 7.1mm (100% $\le 30.0\,\text{mm}$, max 25.1mm), slip mean 15.49mm (max 46.86mm $\le 55.0\,\text{mm}$), normal force mean 8.9N (max 13.6N $\le 30.0\,\text{N}$).
     - First-try acceptance rate: 91% (50 accepted / 55 attempts).
     - Calibrated reachable workspace: $x \in [0.28, 0.325]\,\text{m}, y \in [-0.02, 0.06]\,\text{m}$ ($r \le 0.33\,\text{m}$).
     - Calibrated eye-in-hand `wrist_cam`: `pos="-0.065 0 0.008"`, framing fingers, cube, and receptacle.
   - `data/nominal_val_v1`: 10 accepted episodes, 3,330 frames @ 50 Hz.
     - SHA-256: `ba7c807507eac80a18c98fa22a5dadad72efc79dd90ab5a0bb4c9e93435218aa`.
     - Seeds: 2000–2010 (independent from train seeds).
     - Receptacle distance mean 6.8mm, 0 boundary decode errors.
   - PyAV video decode across boundaries verified with 0 errors.
   - 0.0% joint limit saturation across all active arm joints.
   - HUD inspection video: `outputs/dataset_inspection/inspection_ep0.mp4`.
7. **Test Suite & Code Hygiene:**
   - Unit tests passing 100% (`pytest tests/ -v`: 25/25 passed).
   - Linting clean (`ruff check .`: 0 errors).
8. **Dual ACT Policy Training & Invariants (Phase 2 Part 2):**
   - True stock LeRobot 0.6.1 architecture: Single shared ResNet-18 visual backbone, 4 encoder layers, 1 decoder layer, 4 VAE encoder layers, $d_{\text{model}}=512$, $n_{\text{heads}}=8$, $d_{\text{feedforward}}=3200$, $\dim(z)=32$, $\beta=10.0$.
   - **ACT-B** (Unconditioned Baseline): 51,573,639 parameters, 602 encoder tokens (`top` 300 + `wrist` 300 + `state` 1 + `latent` 1).
   - **ACT-G** (Goal-Conditioned): 51,581,319 parameters, 603 encoder tokens (`top` 300 + `wrist` 300 + `state` 1 + `latent` 1 + `env` 1).
   - Exact parameter delta: 7,680 ($7,168$ for `Linear(13, 512)` + $512$ for the 1D positional embedding table expansion).
   - Memory profile on RTX 3070 8GB: 3,242.3 MB peak VRAM at micro-batch 8 + grad accum 2 (effective batch size 16), leaving ~4.95 GB headroom.
   - Normalization: `VISUAL -> MEAN_STD`, `STATE -> MEAN_STD`, `ACTION -> MEAN_STD`, `ENV -> IDENTITY` (protects against variance collapse on zero-frequency recovery primitive).
   - 100-step smoke tests completed cleanly for both `act-b` and `act-g`, validating loss decrease, gradient scaling, and checkpoint saving.
   - Non-privileged evaluation: `ObservableGoalProvider` in `scripts/eval_policy.py` provides RGB-D unprojected goal snapshots for ACT-G rollouts with zero privileged simulator leakage.

---

## 4. Architectural Documents & Living Artifacts

- [`ARCHITECTURE.md`](./ARCHITECTURE.md): Master high-level evergreen architecture map.
- [`ROADMAP.md`](./ROADMAP.md): Gate 0 + 4-phase agile implementation roadmap with inputs, outputs, constraints, and success criteria.
- [`AGENTS.md`](./AGENTS.md): Operational constitution (Karpathy Rules, verification ladder, physics safety, one-line pointers).
- [`docs/dossier_01_lerobot_agentic.md`](./docs/dossier_01_lerobot_agentic.md): Fully audited student-ready dossier with Section 6 Architectural Reference Scaffold.
- [`docs/subsystem/`](./docs/subsystem/): Comprehensive subsystem design documents.
- [`docs/algorithm/`](./docs/algorithm/): Academic literature mathematical derivations (ACT, Diffusion, SmolVLA, VoxPoser).
- [`docs/DL-pipeline/`](./docs/DL-pipeline/): Deep learning training specifications, dataset formulations, and hyperparameters.

---

## 5. Current Task & Next Actionable Steps

1. **Dual ACT Training Pipeline & Invariant Test Suite Complete ✅:**
   - Implemented `scripts/train_policy.py` supporting `--policy-type {act-b, act-g}`, AMP FP16, parameter-grouped optimizer (10x lower backbone LR), and cosine annealing.
   - Implemented `scripts/eval_policy.py` with `ObservableGoalProvider`, Wilson Score 95% CI, and HUD video export.
   - Built `tests/test_policy_training.py` verifying exact parameter counts, delta 7,680, queue depletion over 10 steps, `policy.reset()`, and AMP training steps (25/25 tests passing).
   - Executed 100-step smoke tests for both ACT-B and ACT-G with ~3.8 GB peak VRAM on RTX 3070.

2. **Immediate Next Step:**
   - Launch production 10,000-step training runs:
     - `outputs/checkpoints/act_b_nominal_v1/`
     - `outputs/checkpoints/act_g_nominal_v1/`
   - Run closed-loop MuJoCo rollout evaluation across validation seeds 2000–2009.

