# Subsystems Architecture Index

This directory provides comprehensive, deep-dive architectural specifications for each core subsystem of **`lerobot-agentic`**.

---

## 1. Subsystems Master Index

| Subsystem | Document | Operating Cadence | Primary Technology | Core Function |
| :---: | :--- | :---: | :--- | :--- |
| **01** | [**Cognitive Supervisory Tier**](./01_cognitive_supervisory_tier.md) | ~0.5–2 Hz Async | Google `gemini-robotics-er-2-preview`, Pydantic | Semantic task decomposition, 2D normalized bounding box regression $[0, 1000]$, visual anomaly detection, closed-loop replanning. |
| **02** | [**Visuomotor Policy Tier**](./02_visuomotor_policy_tier.md) | 50 Hz | Hugging Face LeRobot (`ACTPolicy`, `SmolVLA-450M`, Diffusion) | Action chunking ($K = 50$, 1.0 s horizon), CVAE transformer architecture, dual-camera fusion, EMA temporal ensembling, quintic minimum-jerk trajectory fallback. |
| **03** | [**Physics Simulation Tier**](./03_physics_simulation_tier.md) | 500 Hz / 50 Hz | DeepMind MuJoCo 3.x, EGL Headless GPU rendering | 6-DoF arm + parallel gripper physics, contact dynamics, safe joint addressing (`jnt_qposadr`), overhead and wrist camera mount points. |
| **04** | [**Dataset & Evaluation Tier**](./04_dataset_and_evaluation_tier.md) | Asynchronous / Batch | Hugging Face `LeRobotDataset` v2.0, Parquet, MP4, OpenCV HUD | Synthetic expert demonstration harvester ($\approx 500\,\text{FPS}$), automated benchmarks (GSR, PR, TTC, Jerk $\mathcal{J}$), telemetry HUD video recording. |

---

## 2. Cross-Tier Interfaces & Data Contracts

The system operates across cleanly defined inter-tier boundaries:

```mermaid
flowchart TD
    subgraph CognitiveTier["Tier 1: Cognitive Supervisory Tier (1–2 Hz)"]
        Gemini["Google Gemini Robotics ER"]
        Schema["SpatialGroundingPlan<br>(Pydantic Schema)"]
        Gemini --> Schema
    end

    subgraph PolicyTier["Tier 2: Visuomotor Policy Tier (50 Hz)"]
        ACT["LeRobot ACT / SmolVLA / Fallback"]
        Ensemble["EMA Temporal Ensemble"]
        ACT --> Ensemble
    end

    subgraph SimTier["Tier 3: Physics Simulation Tier (500 Hz / 50 Hz)"]
        MuJoCo["MuJoCo 3.x Engine"]
        Sensors["EGL Cameras (Top, Wrist) + Proprio"]
        MuJoCo --> Sensors
    end

    subgraph EvalTier["Tier 4: Dataset & Evaluation Tier"]
        Harvester["Expert Demonstration Harvester"]
        DatasetHub["LeRobotDataset v2.0 (Parquet + MP4)"]
        Bench["Benchmark Evaluator (GSR, PR, TTC, Jerk)"]
        Harvester --> DatasetHub
        DatasetHub --> ACT
        MuJoCo -.-> Bench
    end

    %% Cross-Tier Dataflow
    Sensors -- "Overhead RGB (480x640x3) @ 1–2 Hz" --> Gemini
    Schema -- "Sub-Goal & Bounding Box [0, 1000]" --> ACT
    Schema -- "should_halt (Emergency Abort)" --> PolicyTier
    Sensors -- "Dual RGB (Top+Wrist) + Proprio (7-DoF) @ 50 Hz" --> ACT
    Ensemble -- "Target Joint Position Setpoints (7-DoF) @ 50 Hz" --> MuJoCo
```

---

## 3. Cross-Tier Data Schemas

### 3.1 Cognitive $\to$ Policy Contract (`SpatialGroundingPlan`)
Defined in [`src/lerobot_agentic/cognitive/schemas.py`](file:///home/aivise/Documents/antigravity/lerobot-agentic/src/lerobot_agentic/cognitive/schemas.py):
- `sub_goal: Literal[...]`: Active state directive (`reach`, `grasp`, `lift`, `transport`, `recover`, etc.).
- `target_object: str`: Semantic object name (`red_cube`).
- `target_box_2d: List[int]`: 4-element normalized bounding box $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ in $[0, 1000]$.
- `destination_box_2d: Optional[List[int]]`: Destination receptacle bounding box in $[0, 1000]$.
- `task_progress: Literal[...]`: Task lifecycle state (`in_progress`, `completed`, `failure_detected`).
- `requires_replanning: bool`: Anomaly flag indicating displacement or grasp failure.
- `replan_id: int`: Monotonically increasing counter for edge-triggered recovery.
- `confidence_score: float`: Affordance confidence in $[0.0, 1.0]$.
- `should_halt: bool`: Emergency abort flag.
- `reasoning: Optional[str]`: Chain-of-thought rationale.

### 3.2 Policy $\to$ Simulation Contract (`Action Chunk`)
Emitted by [`src/lerobot_agentic/policy/executor.py`](file:///home/aivise/Documents/antigravity/lerobot-agentic/src/lerobot_agentic/policy/executor.py):
- Shape: $\mathbf{A}_t \in \mathbb{R}^{50 \times 7}$ (lookahead horizon $K = 50$ steps).
- Dispatched to actuators at each 50 Hz step as a 7-element vector $a_t = [q_1, q_2, q_3, q_4, q_5, q_6, q_7]$.
- Values: $q_{1\dots6}$ target joint positions in radians; $q_7$ target parallel finger displacement in meters.

### 3.3 Simulation $\to$ Policy & Cognitive Contract (`Observation Dict`)
Returned by [`src/lerobot_agentic/sim/env.py`](file:///home/aivise/Documents/antigravity/lerobot-agentic/src/lerobot_agentic/sim/env.py):
- `rgb`: Overhead camera frame `(480, 640, 3)` `uint8`.
- `proprioception`: 7-DoF joint state `(7,)` `float32` resolved via `jnt_qposadr`.
- `cube_pos`: Global 3D Cartesian coordinates `[x, y, z]` of target cube.
- `palm_pos`: Global 3D Cartesian coordinates `[x, y, z]` of gripper palm.

---

## 4. Multi-Rate Timing & Cadence Synchronization

| Tier | Update Rate | Period ($\Delta t$) | Compute Location | Synchronization Mechanism |
| :--- | :---: | :---: | :--- | :--- |
| **Cognitive Supervisory** | 1–2 Hz | $500\text{--}1000\,\text{ms}$ | Cloud API / Asynchronous Thread | Emits updated sub-goals and bounding boxes non-blocking every 50 simulation steps. |
| **Visuomotor Policy** | 50 Hz | $20\,\text{ms}$ | Local GPU (RTX 3070 / 4090) | Generates action chunks; smooths overlapping horizons via EMA temporal ensembling. |
| **Physics Simulation** | 500 Hz | $2\,\text{ms}$ | CPU Core / EGL Offscreen Context | Executes 10 numerical integration substeps (`mj_step`) per 20 ms policy step. |
| **Dataset & Evaluation** | Batch / Rollout | Episode-based | Disk / PyTorch / OpenCV | Logs Parquet/MP4 streams, aggregates GSR, PR, TTC, and Jerk metrics. |
