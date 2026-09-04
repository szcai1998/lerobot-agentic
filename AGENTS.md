# AGENTS.md: Development & Architectural Guidelines for `lerobot-agentic`

Welcome to **`lerobot-agentic`**, a state-of-the-art Embodied AI manipulation engine bridging Google Gemini Robotics Embodied Reasoning APIs with Hugging Face LeRobot visuomotor policies (`ACTPolicy`, `SmolVLA-450M`) and DeepMind MuJoCo physics simulation.

---

## 🏗️ Architecture Topology

The system operates across a dual-rate hierarchical control loop:

1. **Cognitive Supervisory Tier (1–2 Hz)**
   - **Engine:** Google Gemini Robotics API (`gemini-robotics-er-2-preview` / `gemini-2.0-flash`).
   - **Responsibility:** Multi-modal spatial perception, zero-shot 2D bounding box detection `[ymin, xmin, ymax, xmax]` normalized to `[0, 1000]`, natural language task decomposition into atomic sub-goals, and visual anomaly detection.
   - **Module:** `lerobot_agentic.cognitive`

2. **Visuomotor Action Chunking Policy (50 Hz)**
   - **Engine:** Hugging Face LeRobot (`ACTPolicy` / `SmolVLA-450M` flow-matching) with minimum-jerk affordance interpolation.
   - **Responsibility:** Receives RGB camera frames and 7-DoF proprioceptive joint angles, outputs action chunks $A_t \in \mathbb{R}^{K \times 7}$ ($K=50$ steps, 1-second horizon).
   - **Module:** `lerobot_agentic.policy`

3. **Deterministic Physics Simulation (500 Hz internal, 50 Hz control)**
   - **Engine:** DeepMind MuJoCo 3.x with headless EGL rendering (`MUJOCO_GL=egl`).
   - **Responsibility:** Simulates 6-DoF arm, parallel finger gripper, overhead/wrist RGB cameras, and contact physics.
   - **Module:** `lerobot_agentic.sim`

---

## ⚙️ Environment Variables

- `GEMINI_API_KEY`: API key for Gemini Robotics ER multimodal cognitive planning. (If omitted, the agent gracefully falls back to autonomous affordance tracking).
- `MUJOCO_GL`: Rendering backend. Default is `egl` for headless Linux / GPU acceleration.

---

## 🚀 Development Quickstart

```bash
# 1. Setup virtual environment with uv
uv venv .venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install -e ".[dev]"

# 3. Run unit tests
pytest tests/

# 4. Execute a simulation rollout
python scripts/run_rollout.py --steps 150 --video --goal "Grasp the red cube and lift it into the workspace"
```

---

## 🛡️ Coding Standards & Conventions

- **Physics Kinematics Safety:** Never assume joint indices map directly to `qpos[:7]`. Freejoints alter the indexing offset. Always resolve joint names via `model.jnt_qposadr[model.joint(name).id]`.
- **Typing & Validation:** All inter-agent and cognitive outputs must be strongly typed using Pydantic models in `lerobot_agentic.cognitive.schemas`.
- **EGL Headless Support:** Ensure `MUJOCO_GL=egl` is exported before initializing any OpenGL or MuJoCo context to prevent headless X11 crashes.
