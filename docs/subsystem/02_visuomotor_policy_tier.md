# Subsystem 02: Visuomotor Policy Tier

The **Visuomotor Policy Tier** forms the high-frequency continuous control engine of `lerobot-agentic`. Operating at **50 Hz** ($\Delta t = 20\,\text{ms}$), this subsystem receives multi-camera RGB visual streams and robot proprioception, conditioned on semantic directives and spatial affordances from the Cognitive Supervisory Tier, to emit millimeter-accurate joint target trajectories.

---

## 1. Architectural Role & Execution Principles

Standard step-by-step Behavioral Cloning (BC) suffers from severe compounding errors (covariate shift): a minor deviation at timestep $t$ shifts the robot into unseen states, leading to catastrophic failure. The Visuomotor Policy Tier mitigates this through **Action Chunking with Transformers (ACT)**, **Visuomotor Diffusion**, and **Continuous Flow-Matching (SmolVLA)**.

Rather than predicting a single scalar action $a_t \in \mathbb{R}^7$, the policy predicts an **Action Chunk** $\mathbf{A}_t \in \mathbb{R}^{K \times 7}$ containing $K = 50$ consecutive future joint target positions:
$$\mathbf{A}_t = \begin{bmatrix} a_t \\ a_{t+1} \\ \vdots \\ a_{t+K-1} \end{bmatrix} \in \mathbb{R}^{50 \times 7}$$

At a 50 Hz control frequency, $K = 50$ corresponds to a lookahead horizon of **1.0 second**, enabling fluid, non-hesitant physical trajectories.

```
+-------------------------------------------------------------------------------------------------+
|                                    VISUOMOTOR POLICY TIER (50 Hz)                               |
|                                                                                                 |
|   Overhead RGB (480x640x3)    Wrist RGB (480x640x3)    Proprioception State (7-DoF qpos)        |
|               │                         │                               │                       |
|               ▼                         ▼                               ▼                       |
|   ┌──────────────────────┐  ┌──────────────────────┐         ┌──────────────────────┐           |
|   │ Vision Backbone      │  │ Vision Backbone      │         │ Linear Projector     │           |
|   │ (ResNet-18 / SigLIP) │  │ (ResNet-18 / SigLIP) │         │ (7 -> d_model)       │           |
|   └──────────┬───────────┘  └──────────┬───────────┘         └──────────┬───────────┘           |
|              │                         │                                │                       |
|              └───────────────────┬─────┴────────────────────────────────┘                       |
|                                  ▼                                                              |
|               ┌──────────────────────────────────────┐                                          |
|               │ Observation Token Sequence           │                                          |
|               │ [Top_tokens, Wrist_tokens, State]    │                                          |
|               └──────────────────┬───────────────────┘                                          |
|                                  │                                                              |
|        ┌─────────────────────────┴─────────────────────────┐                                    |
|        ▼ (During Training)                                 ▼ (During Inference)                 |
| ┌───────────────────────────┐                       ┌─────────────────────────────┐             |
| │ CVAE Encoder (BERT-like)  │                       │ Standard Normal Prior       │             |
| │ q_ϕ(z | A_t, s_t)         │                       │ z ~ 𝒩(0, I) ∈ ℝ^32          │             |
| └─────────────┬─────────────┘                       └──────────────┬──────────────┘             |
|               │                                                    │                            |
|               └─────────────────────────┬──────────────────────────┘                            |
|                                         ▼                                                       |
|                       ┌─────────────────────────────────────┐                                   |
|                       │ Transformer Decoder (Action Chunk)  │                                   |
|                       │ 50 Learnable Query Tokens           │                                   |
|                       └──────────────────┬──────────────────┘                                   |
|                                          ▼                                                      |
|                       ┌─────────────────────────────────────┐                                   |
|                       │ Predicted Chunk A_t ∈ ℝ^(50 x 7)    │                                   |
|                       └──────────────────┬──────────────────┘                                   |
|                                          ▼                                                      |
|                       ┌─────────────────────────────────────┐                                   |
|                       │ Temporal Action Ensembling (EMA)    │                                   |
|                       └──────────────────┬──────────────────┘                                   |
+------------------------------------------│------------------------------------------------------+
                                           ▼
                            To MuJoCo Actuators (50 Hz ctrl)
```

---

## 2. Policy Engines: ACT, SmolVLA, and Diffusion

The architecture supports multiple plug-and-play visuomotor backends through a unified interface (`VisuomotorPolicyExecutor`):

### 2.1 Action Chunking with Transformers (ACTPolicy)
- **Reference**: Zhao et al. (RSS 2023).
- **Core Mechanism**: Conditional Variational Autoencoder (CVAE) transformer. An encoder synthesizes style latent $z \in \mathbb{R}^{32}$ from demonstration chunks, while a decoder generates $K = 50$ actions from visual tokens, robot state, and $z$.
- **Strengths**: Extreme sample efficiency ($\sim 50$ demonstrations sufficient for complex tasks), sub-20 ms latency, highly deterministic trajectory generation.
- **VRAM Footprint**: $\approx 1.4\,\text{GB}$ FP16 inference, $\approx 4.8\,\text{GB}$ training on consumer GPUs (NVIDIA RTX 3070 / 4090).

### 2.2 SmolVLA-450M (Flow-Matching Foundation VLA)
- **Reference**: Hugging Face LeRobot (2025).
- **Core Mechanism**: Fused multimodal vision backbone (SigLIP) + compact language model (SmolLM-360M) coupled to an Optimal Transport Conditional Flow Matching (OT-CFM) action expert head.
- **Strengths**: Direct natural language conditioning with generalized visual representations, running at 50 Hz via 2-step Heun/Euler ODE solvers.
- **VRAM Footprint**: $\approx 1.2\,\text{GB}$ FP16 inference.

### 2.3 Visuomotor Diffusion Policy
- **Reference**: Chi et al. (RSS 2023, IJRR 2024).
- **Core Mechanism**: DDPM/DDIM stochastic denoising over 1D temporal convolutional U-Nets or Diffusion Transformers (DiT).
- **Strengths**: Excels at highly multi-modal distributions (e.g., bi-stable obstacle avoidance where going left or right is equally valid).

---

## 3. Mathematical Formulation of ACT (CVAE Transformer)

### 3.1 CVAE Evidence Lower Bound (ELBO)
The CVAE models the conditional distribution of action chunks $p_\theta(\mathbf{A}_t \mid s_t)$ by maximizing the ELBO:
$$\log p_\theta(\mathbf{A}_t \mid s_t) \ge \mathbb{E}_{q_\phi(z \mid \mathbf{A}_t, s_t)}\left[ \log p_\theta(\mathbf{A}_t \mid z, s_t) \right] - \beta \cdot D_{\mathrm{KL}}\left( q_\phi(z \mid \mathbf{A}_t, s_t) \,\parallel\, p(z) \right)$$

where:
- $\mathbf{A}_t \in \mathbb{R}^{K \times d_a}$ is the ground-truth action trajectory chunk.
- $s_t = (o_t^{\text{top}}, o_t^{\text{wrist}}, \bar{a}_t)$ is the multi-modal state observation.
- $q_\phi(z \mid \mathbf{A}_t, s_t)$ is the approximate posterior encoder outputting mean $\boldsymbol{\mu}_z$ and log-variance $\log \boldsymbol{\sigma}_z^2$.
- $p(z) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ is the standard multivariate isotropic Gaussian prior.
- $\beta \in [10, 100]$ is the KL divergence regularization weighting schedule.

### 3.2 Closed-Form Loss Functions
The total loss minimized during local training is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot \mathcal{L}_{\mathrm{KL}}$$

The reconstruction loss uses the $L_1$ norm over all chunk steps and degrees of freedom:
$$\mathcal{L}_{\text{reconstruction}} = \frac{1}{K \cdot d_a} \sum_{k=0}^{K-1} \sum_{j=1}^{d_a} \left| a_{t+k, j} - \hat{a}_{t+k, j} \right|$$

The KL divergence for diagonal multivariate Gaussians evaluates analytically:
$$\mathcal{L}_{\mathrm{KL}} = D_{\mathrm{KL}}\left( \mathcal{N}(\boldsymbol{\mu}_z, \boldsymbol{\Sigma}_z) \,\parallel\, \mathcal{N}(\mathbf{0}, \mathbf{I}) \right) = -\frac{1}{2} \sum_{i=1}^{d_z} \left( 1 + \log(\sigma_{z, i}^2) - \mu_{z, i}^2 - \sigma_{z, i}^2 \right)$$

During inference, the encoder is bypassed entirely: $z$ is either sampled from $\mathcal{N}(\mathbf{0}, \mathbf{I})$ or set deterministically to the prior mean $\mathbf{0}$ for maximum repeatability.

---

## 4. Multi-Camera Visual Feature Extraction & Fusion

The perception pipeline processes two complementary viewpoints to resolve depth ambiguity and self-occlusion:

```
Overhead Camera (480x640x3)     ──► [ResNet-18 Backbone] ──► 1x1 Conv (512 -> 512) ──► Tokens (H' x W' x 512)
Wrist Camera (480x640x3)        ──► [ResNet-18 Backbone] ──► 1x1 Conv (512 -> 512) ──► Tokens (H' x W' x 512)
Proprioception qpos (7-DoF)     ──► [Linear Projection]  ─────────────────────────────► Token  (1 x 512)
Goal Vector (13-DoF env_state)  ──► [Linear Projection]  ─────────────────────────────► Token  (1 x 512)
CVAE Latent Vector z (32)       ──► [Linear Projection]  ─────────────────────────────► Token  (1 x 512)
```

1. **Overhead Camera (`observation.images.top`)**:
   - Resolution: $480 \times 640 \times 3$, normalized to $[0, 1]$.
   - Spatial context: Global table frame, workspace boundaries, manipuland coordinates, and target drop receptacle.
2. **Wrist Camera (`observation.images.wrist`)**:
   - Resolution: $480 \times 640 \times 3$, normalized to $[0, 1]$.
   - Spatial context: Eye-in-hand alignment, micro-adjustments during final descent, finger-object gap estimation.
3. **Proprioceptive State (`observation.state`)**:
   - $7$-dimensional vector: 6 arm joint angles $[q_1, \dots, q_6]$ (radians) + 1 gripper slide displacement $q_7$ (meters).
   - Normalized using dataset statistics $(\boldsymbol{\mu}_s, \boldsymbol{\sigma}_s)$ stored in `meta/stats.json`.
4. **Goal Conditioning Vector (`observation.environment_state`)**:
   - 13-dimensional vector: $[\mathbf{p}_{\text{target}}^{3D}, \mathbf{p}_{\text{dest}}^{3D}, \mathbf{e}_{\text{subgoal}}]$ (`FeatureType.ENV`), where $\mathbf{e}_{\text{subgoal}} \in \mathbb{R}^7$ is a 7-class one-hot encoding corresponding to the 7 canonical primitives: `["reach", "grasp", "lift", "transport", "place", "retreat", "recover"]`.
   - Ingested via stock LeRobot ACT's native `encoder_env_state_input_proj`, conditioning the CVAE transformer on cognitive subgoals and metric 3D affordances without custom library forks.

---

## 5. Inference Execution Modes & Action Queuing

### 5.1 Primary Benchmark Mode: Queue / Receding Horizon
In the primary benchmark configuration:
- Parameters: `chunk_size = 50`, `n_action_steps = 10`, `temporal_ensemble_coeff = None`.
- **Temporal Decoupling**: ACT predicts a 50-step action chunk (1.0 s horizon). The control loop consumes $n_{\text{action\_steps}} = 10$ actions at 50 Hz before triggering the next model inference, yielding a nominal inference cadence of $\approx 5\,\text{Hz}$ ($\Delta t = 200\,\text{ms}$).
- **Edge-Triggered Queue Reset**: When a supervisory recovery event is asserted (`replan_id` increments), calling `policy.reset()` instantly flushes stale buffered actions, ensuring newly generated corrective trajectories execute with zero latency.

### 5.2 Optional Ablation Mode: Temporal Action Ensembling (EMA)
As an optional secondary ablation, `lerobot-agentic` supports continuous Exponential Moving Average (EMA) Temporal Ensembling:
- At each 50 Hz control step $t$, the policy performs inference, obtaining a new prediction chunk. The executed action $\bar{a}_t$ is computed as an exponentially-weighted blend of all overlapping historical chunks predicting for step $t$:

$$\bar{a}_t = \frac{\sum_{i=0}^{\min(t, K-1)} w_i \cdot \hat{a}_{t-i}[i]}{\sum_{i=0}^{\min(t, K-1)} w_i}$$

where $w_i = \exp(-m \cdot i)$ is the exponential decay weight ($m \in [0.01, 0.1]$).

---

## 6. Pre/Post Processor Pipeline Architecture

In accordance with modern LeRobot 0.6+ pipeline discipline, normalization and tensor transformations are encapsulated in explicit pre/post processor pipelines rather than hardcoded in the policy forward pass:

```
Raw MuJoCo Observation
        │  Top RGB: (480, 640, 3) in [0, 255] uint8
        │  Wrist RGB: (480, 640, 3) in [0, 255] uint8
        │  Proprioception: (7,) float32
        │  Environment State: (13,) float32
        ▼
Environment Processor (Zero Copy / Adapter)
        │  Unbatched Tensors:
        │  - observation.images.top: (3, 480, 640) float32 in [0, 1]
        │  - observation.images.wrist: (3, 480, 640) float32 in [0, 1]
        │  - observation.state: (7,) float32
        │  - observation.environment_state: (13,) float32
        ▼
LeRobot Policy Preprocessor Pipeline (restored via make_pre_post_processors)
        │  - AddBatchDimensionProcessorStep -> (1, ...)
        │  - DeviceProcessorStep -> GPU VRAM
        │  - NormalizerProcessorStep (Mean/Std or Min/Max from stats.json)
        ▼
ACT select_action(batch)
        │  Outputs normalized action: (1, 7)
        ▼
LeRobot Postprocessor Pipeline
        │  - UnnormalizerProcessorStep -> Physical joint units (rad, m)
        │  - DeviceProcessorStep -> Host CPU
        │  - RemoveBatchDimensionProcessorStep -> (7,)
        ▼
Action / Joint Adapter
        │  - Joint command clamping: [-pi, pi] for arm joints, [-0.025, 0.025] for gripper finger
        ▼
MuJoCo Actuator Positions (50 Hz)
```

---

## 7. Implementation Reference

The Visuomotor Policy Tier is implemented in:
- **`src/lerobot_agentic/policy/executor.py`**: `GoalConditionedACTPolicyExecutor` orchestrating checkpoint restoration (`make_pre_post_processors`), observation serialization, 13-DoF goal vector encoding, queue-based receding horizon execution, and dynamic recovery queue flushing (`policy.reset()`).
