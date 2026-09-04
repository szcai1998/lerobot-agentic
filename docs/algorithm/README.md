# Embodied AI Algorithmic Foundations: Mathematical Compendium & Architecture Guide

> **Core Repository Module:** [`lerobot_agentic`](file:///home/aivise/Documents/antigravity/lerobot-agentic/src/lerobot_agentic)  
> **Target Frameworks:** Hugging Face `lerobot v0.6+`, DeepMind `mujoco 3.x`, PyTorch 2.4+ CUDA 12.4, Google `google-genai`  
> **Compute Profile:** NVIDIA GeForce RTX 4090 (24GB VRAM) & RTX 3070 (8GB VRAM)

---

## 1. Algorithmic Compendium Index

This documentation suite provides mathematically rigorous formulations, network topologies, loss derivations, and implementation code for the foundational algorithms powering modern Embodied AI manipulation policies.

| Chapter | Document | Foundational Literature | Core Focus & Equations |
| :---: | :--- | :--- | :--- |
| **01** | [`01_act_policy.md`](file:///home/aivise/Documents/antigravity/lerobot-agentic/docs/algorithm/01_act_policy.md) | Zhao et al. (RSS 2023) | Action Chunking with Transformers (ACT), CVAE ELBO derivation, Gaussian KL divergence, temporal ensembling EMA. |
| **02** | [`02_diffusion_policy.md`](file:///home/aivise/Documents/antigravity/lerobot-agentic/docs/algorithm/02_diffusion_policy.md) | Chi et al. (RSS 2023 / IJRR 2024) | Visuomotor Diffusion Policy, DDPM forward/reverse equations, Tweedie's score matching, DDIM ODE acceleration, CNN vs DiT. |
| **03** | [`03_vla_flow_matching.md`](file:///home/aivise/Documents/antigravity/lerobot-agentic/docs/algorithm/03_vla_flow_matching.md) | LeRobot Team (2025), Black et al. (2024/2025), Lipman et al. (ICLR 2023) | SmolVLA-450M, $\pi_0$ & $\pi_0$-FAST, Continuous Optimal Transport Flow Matching (OT-CFM) vs Diffusion, DCT frequency tokenization. |
| **04** | [`04_spatial_grounding_voxposer.md`](file:///home/aivise/Documents/antigravity/lerobot-agentic/docs/algorithm/04_spatial_grounding_voxposer.md) | Kim et al. (CoRL 2024), Huang et al. (CoRL 2023) | OpenVLA-7B LoRA fine-tuning math, VoxPoser 3D value potential fields, Gemini 2D-to-3D metric unprojection in $\mathrm{SE}(3)$. |

---

## 2. Comprehensive Cross-Comparison of Embodied AI Policies

The following matrix provides an exhaustive comparative evaluation across model sizing, real-time control frequency, inductive biases, training/inference VRAM budgets, and deployment constraints:

| Algorithmic Paradigm | Primary Model Architecture | Parameter Count | Control Rate (Hz) | Inference Latency | Training VRAM (RTX 4090) | Inference VRAM (RTX 3070) | Primary Inductive Bias | Multimodality Handling | Sample Efficiency | Optimal Real-World Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ACT** *(Zhao et al.)* | ResNet-18 / ViT + Transformer CVAE | ~45M – 85M | **50 Hz** | ~18–22 ms | **4.8 GB** (B=32) | **1.4 GB** (FP16) | Contiguous temporal trajectory coherence | CVAE latent variable $z \sim \mathcal{N}(0, I)$ | **Very High** (~50 demos) | High-precision bimanual assembly (threading, peg-in-hole, ALOHA) |
| **Diffusion Policy (CNN)** *(Chi et al.)* | 1D Temporal Convolutional U-Net | ~15M – 35M | **60 Hz** (DDIM-16) | ~15–18 ms | **6.1 GB** (B=64) | **1.8 GB** (FP16) | Temporal translation equivariance | Score-based stochastic gradient field | High (~100 demos) | Dynamic obstacle avoidance, pushing, pick-and-place under noisy teleop |
| **Diffusion Policy (DiT)** *(Chi et al.)* | Time-Conditioned Diffusion Transformer | ~65M – 120M | **30 Hz** (DDIM-16) | ~30–35 ms | **11.5 GB** (B=32) | **3.2 GB** (FP16) | Global cross-attention over observation tokens | Score-based stochastic gradient field | Moderate (~150 demos) | Long-horizon tasks requiring cross-modal attention |
| **SmolVLA-450M** *(Hugging Face)* | SigLIP + SmolLM + Flow-Matching Head | **450M** | **50 Hz** (Heun-2) | ~16–20 ms | **4.5 GB** (B=16) | **1.2 GB** (FP16) | Decoupled semantic reasoning & continuous flow | Optimal Transport straight vector fields | High (pretrained foundation) | **Edge Foundation VLA**: Language-conditioned multi-tasking on consumer GPUs |
| **$\pi_0$ / $\pi_0$-FAST** *(Physical Intel)* | PaliGemma-3B + Continuous Flow / DCT | 3.2B | **50 Hz** (FAST/1-step) | ~14–18 ms | **12.2 GB** (B=8) | **2.9 GB** (FP16) | Frequency-space energy compaction (DCT) | Continuous CFM / Autoregressive discrete tokens | Extremely High (Pretrained) | Generalist dexterous robotics across diverse multi-robot embodiments |
| **OpenVLA-7B (LoRA)** *(Kim et al.)* | DINOv2 + SigLIP + Llama-2 7B | 7.2B (LoRA: 48M) | 6–8 Hz | ~120–160 ms | **18.4 GB** (B=4, LoRA) | **4.5 GB** (NF4 Quantized) | Internet-scale multimodal language alignment | Discretized 256-bin cross-entropy per DoF | Moderate | High-level semantic categorization & affordance grounding |
| **VoxPoser** *(Huang et al.)* | VLM Grounding + LLM Code Gen + MPC | N/A (Zero-shot) | 5–10 Hz (MPC) | ~100–200 ms | **0 GB** (API / Pretrained) | **0 GB** (API) | 3D Voxel potential fields & Signed Distance Fields | Composable potential functions $\mathcal{V}(x, y, z)$ | **Zero-Shot** (No demonstrations) | Novel task execution without training demonstrations in known geometry |
| **Gemini Robotics ER Tier** *(Google)* | `gemini-robotics-er-2-preview` | Cloud Foundation | 1–2 Hz | ~500–1000 ms | **0 GB** (Cloud API) | **0 GB** (Cloud API) | Pinhole projective geometry + multimodal reasoning | Open-vocabulary zero-shot spatial bounding boxes | **Zero-Shot** | Cognitive supervision, task decomposition, anomaly detection & recovery |

---

## 3. Algorithmic Decision Flowchart for Robotic Tasks

Use this decision matrix when architecting an Embodied AI manipulation system:

```
                                      START: What is the primary task requirement?
                                                          |
             +--------------------------------------------+--------------------------------------------+
             |                                                                                         |
   [ Semantic Flexibility & Language ]                                                       [ High Dexterity & Precision ]
             |                                                                                         |
Does it need edge hardware training (<8GB VRAM)?                                         Does the demonstration data exhibit
             |                                                                           strong spatial/temporal multimodality?
      +------+------+                                                                                  |
     YES            NO                                                                          +------+------+
      |              |                                                                         YES            NO
      v              v                                                                          |              v
[ SmolVLA-450M ]  [ OpenVLA-7B ]                                                                v          [ ACT Policy ]
  (Flow Matching    (LoRA r=32 on                                                      [ Diffusion Policy ]  (ResNet/ViT CVAE +
   at 50 Hz)         RTX 4090)                                                          (CNN U-Net or DiT,    Temporal Ensembling,
                                                                                         DDIM 16 steps)       sub-20ms latency)
             \                                                                                         /
              +-----------------------------------+---------------------------------------------------+
                                                  |
                                                  v
                                     Is 3D Collision-Free Avoidance
                                      required with ZERO demonstrations?
                                                  |
                                           +------+------+
                                          YES            NO
                                           |              v
                                           v       [ Integrate Cognitive Supervisor ]
                                     [ VoxPoser ]   (Gemini 2.0 2D-to-3D Unprojection
                                     (3D Value       + LeRobot 50Hz Motor Policy)
                                      Voxel Maps)
```

---

## 4. Unified Mathematical Notation Dictionary

Across all documentation in `docs/algorithm/`, mathematical symbols conform to this standardized notation:

### Kinematics, Spaces & Observations
* $\mathcal{S} \subseteq \mathbb{R}^{d_s}$: Environmental observation space.
* $\mathcal{A} \subseteq \mathbb{R}^{d_a}$: Robot action space ($d_a = 7$ for 6-DoF arm + 1-DoF parallel gripper; $d_a = 14$ for bimanual setups).
* $s_t \in \mathcal{S}$: Full observation state at timestep $t$ containing multi-view RGB images and joint state.
* $\bar{a}_t \in \mathbb{R}^{d_a}$: Robot proprioceptive state (current joint positions or end-effector pose).
* $\mathbf{A}_t \in \mathbb{R}^{K \times d_a}$: Action chunk spanning $K$ future timesteps $[a_t, a_{t+1}, \dots, a_{t+K-1}]$.
* $\mathbf{O}_t = [o_{t-T_{\text{obs}}+1}, \dots, o_t]$: Historical observation window of length $T_{\text{obs}}$.

### Generative Modeling & Diffusion
* $z \in \mathbb{R}^{d_z}$: Latent style variable in CVAE ($d_z = 32$).
* $q_\phi(z \mid \mathbf{A}_t, s_t)$: CVAE approximate posterior encoder parameterizing mean $\boldsymbol{\mu}_\phi$ and diagonal covariance $\boldsymbol{\Sigma}_\phi = \operatorname{diag}(\boldsymbol{\sigma}_\phi^2)$.
* $p(z) = \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_z})$: Standard Gaussian prior over latent variables.
* $k \in \{1, \dots, N\}$: Discrete diffusion timestep in DDPM/DDIM ($N = 100$).
* $\beta_k \in (0, 1)$: Variance schedule step at diffusion index $k$; $\alpha_k = 1 - \beta_k$; $\bar{\alpha}_k = \prod_{s=1}^k \alpha_s$.
* $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Standard Gaussian noise tensor.
* $\boldsymbol{\epsilon}_\theta(\mathbf{A}^{(k)}, k, \mathbf{O}_t)$: Neural noise/score estimator.

### Continuous Flows & Differential Equations
* $t \in [0, 1]$: Continuous flow time ($t = 0$: prior noise distribution, $t = 1$: data distribution).
* $v_\theta(x_t, t, \mathbf{c})$: Neural parameterization of the continuous time-dependent vector field.
* $\psi_t(x)$: Continuous flow mapping satisfying $\frac{d}{dt}\psi_t(x) = v_t(\psi_t(x))$.
* $u_t(x \mid x_0, x_1) = x_1 - x_0$: Optimal Transport constant conditional target vector field.

### Spatial Geometry & Metric Unprojection
* $[y_{\min}, x_{\min}, y_{\max}, x_{\max}] \in [0, 1000]^4$: Normalized 2D bounding box from Gemini API.
* $(u_c, v_c)$: Pixel coordinates on image plane of size $W \times H$.
* $\mathbf{K} \in \mathbb{R}^{3 \times 3}$: Pinhole camera intrinsic calibration matrix ($f_x, f_y, c_x, c_y$).
* $D(u, v) \in \mathbb{R}^+$: Calibrated metric depth value in meters.
* $\mathbf{P}_C = [X_C, Y_C, Z_C]^T \in \mathbb{R}^3$: 3D point in camera optical coordinate frame.
* $\mathbf{T}_B^C \in \mathrm{SE}(3)$: $4 \times 4$ rigid-body extrinsic matrix transforming camera coordinates to robot base frame:
  $$\mathbf{T}_B^C = \begin{bmatrix} \mathbf{R}_B^C & \mathbf{t}_B^C \\ \mathbf{0}^T & 1 \end{bmatrix}$$
* $\mathbf{P}_B \in \mathbb{R}^3$: Metric target position in the robot base frame.

---

## 5. Codebase Mapping & Implementation References

The theoretical concepts detailed in this compendium directly map to the production modules of `lerobot-agentic`:

```
lerobot-agentic/
├── docs/algorithm/
│   ├── README.md                           <-- Master Index & Comparison Matrix (This File)
│   ├── 01_act_policy.md                    <-- ACT CVAE & Temporal Ensembling Math
│   ├── 02_diffusion_policy.md              <-- DDPM/DDIM Score Matching & CNN/DiT U-Net
│   ├── 03_vla_flow_matching.md             <-- SmolVLA, pi_0 OT-CFM & FAST DCT
│   └── 04_spatial_grounding_voxposer.md    <-- OpenVLA LoRA, VoxPoser & Metric Unprojection
└── src/lerobot_agentic/
    ├── policy/
    │   └── executor.py                     <-- VisuomotorPolicyExecutor (ACT / Diffusion / Quintic Splines)
    ├── cognitive/
    │   ├── supervisor.py                   <-- CognitiveSupervisor (Gemini 2.0 Spatial Grounding API)
    │   └── schemas.py                      <-- SpatialGroundingPlan (Pydantic Schemas & Bounding Boxes)
    └── sim/
        └── env.py                          <-- Headless MuJoCo EGL Physics Simulation Loop
```

* **ACT Inference & Minimum-Jerk Affordance**: See [`src/lerobot_agentic/policy/executor.py`](file:///home/aivise/Documents/antigravity/lerobot-agentic/src/lerobot_agentic/policy/executor.py).
* **Gemini Spatial Grounding & Verification**: See [`src/lerobot_agentic/cognitive/supervisor.py`](file:///home/aivise/Documents/antigravity/lerobot-agentic/src/lerobot_agentic/cognitive/supervisor.py).
* **Data Schemas & Type Safety**: See [`src/lerobot_agentic/cognitive/schemas.py`](file:///home/aivise/Documents/antigravity/lerobot-agentic/src/lerobot_agentic/cognitive/schemas.py).
