# LeRobot-Agentic 🦾🤖
### Embodied AI Manipulation Engine: Hugging Face LeRobot + DeepMind MuJoCo + Gemini Robotics Embodied Reasoning

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/Hugging_Face-LeRobot-yellow.svg)](https://github.com/huggingface/lerobot)
[![Physics: MuJoCo 3.x](https://img.shields.io/badge/Physics-MuJoCo_3.x-red.svg)](https://mujoco.org/)

**LeRobot-Agentic** is a production-grade Embodied AI framework that combines high-level multimodal semantic intelligence with high-frequency continuous motor control. It bridges **Google Gemini Robotics Embodied Reasoning APIs** with **Hugging Face LeRobot policies (`ACTPolicy`, `SmolVLA-450M`)** and **DeepMind MuJoCo** deterministic simulation.

---

## 🏛️ System Architecture

```text
+-------------------------------------------------------------------------------+
|                        COGNITIVE REASONING TIER (1-2 Hz)                      |
|         Google gemini-robotics-er-2-preview / gemini-2.0-flash (API)          |
|   - Zero-shot spatial grounding & bounding box detection [ymin, xmin, ymax, xmax] |
|   - Task decomposition: "Grasp red cube -> Lift -> Transport to target zone"  |
|   - Visual anomaly detection & closed-loop replanning                         |
+---------------------------------------+---------------------------------------+
                                        | Semantic Sub-goals & Spatial Boxes
                                        v
+-------------------------------------------------------------------------------+
|                       VISUOMOTOR POLICY CONTROLLER (50 Hz)                    |
|             Hugging Face LeRobot (ACTPolicy / SmolVLA-450M Engine)            |
|                                                                               |
|   - Multi-camera RGB feed (overhead + wrist) + 7-DoF Proprioception State     |
|   - Action Chunking with Transformers (K=50 steps, 1-second horizon)          |
|   - Minimum-jerk polynomial affordance smoothing & temporal ensembling        |
+---------------------------------------+---------------------------------------+
                                        | Target Joint Commands (50 Hz)
                                        v
+-------------------------------------------------------------------------------+
|                    PHYSICS SIMULATION & HARDWARE ABSTRACTION                  |
|          DeepMind MuJoCo 3.x (Headless EGL GPU Renderer @ 500 Hz physics)     |
|   - 6-DOF Manipulator + Parallel Gripper + Multi-camera Sensors               |
|   - Open Hardware Ready: SO-100 / SO-101 / LeKiwi Mobile Manipulator          |
+-------------------------------------------------------------------------------+
```

---

## 💻 Hardware Suitability

- **Local Edge Node (Tested):** NVIDIA GeForce RTX 3070 (8GB VRAM Ampere)
  - Seamless MuJoCo EGL headless rendering at >200 FPS.
  - ACTPolicy / SmolVLA-450M low-latency inference (~500MB to 4.5GB VRAM footprint).
- **Server Compute Node:** NVIDIA GeForce RTX 4090 (24GB VRAM Ada Lovelace)
  - Full dataset training & parallel rollout environments.

---

## ⚡ Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/szcai1998/lerobot-agentic.git
cd lerobot-agentic

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install package in editable mode
uv pip install -e ".[dev]"
```

### 2. Set Up Environment Variables (Optional)

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export MUJOCO_GL="egl"
```
*(Note: If `GEMINI_API_KEY` is not exported, the system automatically falls back to autonomous affordance tracking).*

### 3. Run Simulation Rollout

```bash
# Run a 150-step closed-loop manipulation episode and generate video
python scripts/run_rollout.py --steps 150 --video --goal "Grasp the red cube and lift it into the workspace"
```

The output video with HUD telemetry overlays is saved directly into `outputs/videos/rollout.mp4`.

---

## 📂 Repository Structure

```text
lerobot-agentic/
├── AGENTS.md                  # Guidelines for autonomous coding agents
├── README.md                  # Project documentation
├── pyproject.toml             # Modern Python build specification
├── requirements.txt           # Dependency manifest
├── src/
│   └── lerobot_agentic/
│       ├── __init__.py
│       ├── cognitive/         # Gemini Robotics ER supervisor & Pydantic schemas
│       │   ├── schemas.py
│       │   └── supervisor.py
│       ├── policy/            # LeRobot ACTPolicy & action chunking executor
│       │   └── executor.py
│       ├── sim/               # MuJoCo EGL environment & MJCF robot models
│       │   ├── env.py
│       │   └── models/
│       │       └── embodied_arm.xml
│       └── utils/             # HUD telemetry & MP4/GIF video recording
│           └── recorder.py
├── scripts/
│   └── run_rollout.py         # Main entry point for closed-loop rollouts
└── tests/                     # Unit test suites
    ├── test_env.py
    └── test_schemas.py
```

---

## 🗺️ Roadmap & Milestones

- [x] **Milestone 1:** Headless MuJoCo EGL physics harness with 6-DOF arm & camera rendering.
- [x] **Milestone 2:** Cognitive Supervisor with Gemini Robotics ER Pydantic schema validation.
- [x] **Milestone 3:** Action chunking executor with temporal ensembling and minimum-jerk trajectory.
- [ ] **Milestone 4:** Pretrained LeRobot `ACTPolicy` checkpoint integration and dataset recording.
- [ ] **Milestone 5:** Physical hardware teleoperation drivers for SO-100 / SO-101 3D-printed arms.

---

## 📄 License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
