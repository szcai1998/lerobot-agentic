# LeRobot-Agentic 🦾🤖
### Embodied AI Manipulation Engine: Hugging Face LeRobot + DeepMind MuJoCo + Gemini Robotics Embodied Reasoning

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/Hugging_Face-LeRobot-yellow.svg)](https://github.com/huggingface/lerobot)
[![Physics: MuJoCo 3.x](https://img.shields.io/badge/Physics-MuJoCo_3.x-red.svg)](https://mujoco.org/)
[![CUDA: 12.4 / 13.0](https://img.shields.io/badge/CUDA-RTX_3070_8GB-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**LeRobot-Agentic** is a production-grade Embodied AI manipulation platform combining high-level multimodal cognitive reasoning with high-frequency continuous visuomotor policy execution. It bridges **Google Gemini Robotics Embodied Reasoning APIs (`gemini-robotics-er-2-preview`)** with **Hugging Face LeRobot policies (`ACTPolicy`, `SmolVLA-450M`)** inside **DeepMind MuJoCo** deterministic physics simulation.

---

## 🧭 Master Documentation Directory

- 🏛️ **System Architecture**: [`ARCHITECTURE.md`](./ARCHITECTURE.md) — Master evergreen design document, dual-rate control loops, and topological dataflow.
- 🗺️ **Project Roadmap**: [`ROADMAP.md`](./ROADMAP.md) — 5-stage bite-sized agile execution plan with explicit inputs, outputs, constraints, and success gates.
- 🧠 **Operational Memory**: [`MEMORY.md`](./MEMORY.md) — Current system state, verified invariants, hardware budgets, and active tasks.
- 🛡️ **Operational Guidelines**: [`AGENTS.md`](./AGENTS.md) — Karpathy rules, verification ladder, kinematics safety, and AI coding constraints.
- 📦 **Subsystem Deep Dives**: [`docs/subsystem/`](./docs/subsystem/) — In-depth architectural guides for Cognitive, Policy, Simulation, and Dataset tiers.
- 📚 **Algorithm Compendium**: [`docs/algorithm/`](./docs/algorithm/) — Mathematical derivations of ACT CVAE, Diffusion Policy, Continuous Flow Matching, and Spatial Grounding.
- ⚡ **Deep Learning Pipeline**: [`docs/DL-pipeline/`](./docs/DL-pipeline/) — Dataset schemas, CVAE transformer architecture, and RTX 3070 local training guide.

---

## 🏛️ Dual-Rate Control Topology

The architecture decouples high-level semantic reasoning from high-frequency joint actuation:

```text
+-------------------------------------------------------------------------------+
|                    1. COGNITIVE SUPERVISORY TIER (1–2 Hz)                     |
|           Google gemini-robotics-er-2-preview / gemini-2.5-flash              |
|                                                                               |
|   • Multi-modal spatial perception & zero-shot 2D bounding boxes [0, 1000]   |
|   • Natural language goal decomposition (Reach ➔ Grasp ➔ Lift ➔ Place)       |
|   • Visual anomaly detection & closed-loop self-correction replanning         |
|   • Autonomous zero-cloud OpenCV color/contour affordance fallback            |
+---------------------------------------+---------------------------------------+
                                        | Semantic Sub-goals & Bounding Boxes
                                        v
+-------------------------------------------------------------------------------+
|                    2. VISUOMOTOR POLICY CONTROLLER (50 Hz)                    |
|           Hugging Face LeRobot (ACTPolicy / SmolVLA-450M in PyTorch)          |
|                                                                               |
|   • Dual-camera visual inputs (Overhead camera + Gripper wrist camera)        |
|   • 7-DoF Proprioceptive joint angle state vector q ∈ ℝ^7                     |
|   • Action Chunking with Transformers (K=50 steps, 1.0-second horizon)        |
|   • Minimum-jerk polynomial affordance smoothing & EMA temporal ensembling    |
+---------------------------------------+---------------------------------------+
                                        | Target Joint Commands (50 Hz)
                                        v
+-------------------------------------------------------------------------------+
|                    3. DETERMINISTIC PHYSICS SIMULATION (500 Hz)               |
|            DeepMind MuJoCo 3.x (Headless GPU EGL Rendering @ >400 FPS)        |
|                                                                               |
|   • 6-DoF robotic manipulator + 1-DoF parallel gripper                        |
|   • Rigid-body contact dynamics, Coulomb friction, and object mass physics    |
|   • Overhead & eye-in-hand cameras rendered in headless Linux environments    |
+-------------------------------------------------------------------------------+
```

---

## 💻 Hardware Suitability & Compute Topology

| Compute Layer | Target Hardware | Verified Metric / Budget |
| :--- | :--- | :--- |
| **Cognitive Tier** | Google AI Studio Managed API | Zero local VRAM, 1–2 Hz call rate |
| **Physics Simulation** | NVIDIA GeForce RTX 3070 (8GB) | EGL Headless GPU rendering at **>400 FPS** |
| **Motor Policy Inference** | NVIDIA GeForce RTX 3070 (8GB) | **~1.4 GB VRAM**, sub-20ms inference latency |
| **Motor Policy Training** | NVIDIA GeForce RTX 3070 (8GB) | **~2.4 GB VRAM** (50k steps in ~45 min, AMP BF16) |

---

## ⚡ Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/szcai1998/lerobot-agentic.git
cd lerobot-agentic

# Create and activate virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install package with lerobot and development tools
uv pip install -e ".[lerobot,dev]"
```

### 2. Configure Environment (Optional)

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=AIzaSyYourDeveloperKeyHere
MUJOCO_GL=egl
```
*(Note: If `GEMINI_API_KEY` is omitted, the agent seamlessly operates via autonomous local computer vision affordance tracking).*

### 3. Run Simulation Rollout

```bash
# Execute a 100-step closed-loop manipulation episode with live Gemini ER grounding
python scripts/run_rollout.py --steps 100 --video --goal "Grasp the red cube and lift it into the workspace"
```

The resulting video with HUD telemetry overlays is saved to `outputs/videos/rollout.mp4`.

---

## 📂 Repository Structure

```text
lerobot-agentic/
├── AGENTS.md                          # Operational constitution (Karpathy rules, physics safety)
├── ARCHITECTURE.md                    # Master high-level evergreen architecture map
├── ROADMAP.md                         # 5-stage bite-sized agile execution roadmap
├── MEMORY.md                          # Current project status, hardware profile & verified invariants
├── README.md                          # Project overview & documentation index
├── pyproject.toml                     # PEP 517/621 build specification & dependency extras
├── requirements.txt                   # Dependency manifest
├── .env                               # Local secrets (gitignored)
│
├── src/lerobot_agentic/               # Clean Architecture: Separation of Concerns
│   ├── sim/                           # [PHYSICS] MuJoCo simulation environment & MJCF models
│   │   ├── env.py                     # 50 Hz control step, cameras, safe joint addressing
│   │   └── models/embodied_arm.xml    # 6-DOF arm, parallel gripper, table, and cameras
│   ├── cognitive/                     # [REASONING] 1–2 Hz Cognitive Supervisor
│   │   ├── supervisor.py              # Gemini Robotics ER / Flash client + local CV fallback
│   │   └── schemas.py                 # Pydantic SpatialGroundingPlan schemas
│   ├── policy/                        # [MOTOR CONTROL] 50 Hz Visuomotor Policy
│   │   └── executor.py                # ACTPolicy loading, temporal ensembling EMA, chunking
│   ├── dataset/                       # [DATA ENGINE] Expert Trajectory Harvesting
│   │   └── generator.py               # Algorithmic expert solver collecting LeRobotDataset v2
│   └── utils/                         # Telemetry & Visualization
│       └── recorder.py                # Video HUD recorder (bounding boxes, sub-goals, MP4/GIF)
│
├── scripts/                           # Executable CLI Workflows
│   ├── run_rollout.py                 # Closed-loop simulation rollout demonstration
│   ├── record_dataset.py              # Synthetic demonstration collection (LeRobot format)
│   ├── train_policy.py                # Train ACT policy locally on RTX 3070
│   └── evaluate.py                    # Quantitative benchmark evaluation suite
│
├── docs/                              # Comprehensive Documentation Suites
│   ├── subsystem/                     # Subsystem design specifications (01 to 04)
│   ├── algorithm/                     # Academic literature math (ACT, Diffusion, SmolVLA, etc.)
│   └── DL-pipeline/                   # Deep learning architecture, dataset specs & RTX 3070 configs
│
├── tests/                             # Unit Test Suite (PyTest)
│   ├── test_env.py                    # MuJoCo physics, kinematics, and EGL renderer tests
│   └── test_schemas.py                # Pydantic spatial grounding validation tests
│
├── data/                              # Demonstration datasets (LeRobot v2 Parquet + MP4)
└── outputs/                           # Checkpoints, benchmark logs, and rollout videos
```

---

## 📄 License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
