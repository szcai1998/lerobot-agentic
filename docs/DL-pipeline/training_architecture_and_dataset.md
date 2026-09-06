# Deep Learning Architecture & Training Pipeline: `lerobot-agentic`

This document provides a comprehensive technical breakdown of the deep learning architecture, dataset formulation, loss functions, and training pipeline in `lerobot-agentic`, specifically written for machine learning and robotics engineers.

---

## 1. High-Level Compute Division: Frozen VLM vs. Trainable Policy

A frequent question when building agentic robotics systems is: **If Gemini is an API call, what is our neural network learning?**

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    TIER 1: FROZEN MULTIMODAL REASONER (Cloud)                    │
│                 Google Gemini Robotics ER (gemini-robotics-er-2-preview)          │
│                                                                                  │
│  • Nature: Zero-shot, non-trainable foundational VLM (API inference).            │
│  • Input: Static RGB image frame + natural language prompt.                      │
│  • Output: Structured JSON (Spatial bounding boxes, sub-goals, anomaly flags).   │
│  • Role: Solves semantic perception, open-world task decomposition, and safety.  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Semantic directives & target bounding boxes
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   TIER 2: TRAINABLE VISUOMOTOR POLICY (Local PyTorch)            │
│                      ACT (Action Chunking with Transformers)                     │
│                                                                                  │
│  • Nature: End-to-end PyTorch neural network trained locally on your RTX 3070.   │
│  • Input: Continuous multi-camera visual streams (Top + Wrist) + Proprioception. │
│  • Output: Continuous 50 Hz joint angle trajectory chunks (50 x 7 tensor).       │
│  • Role: Solves high-frequency physical dynamics, contact, and trajectory math.  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

The VLM tells the robot **what to do and where targets are in image space**, but it **cannot drive motors**:
1. LLM/VLMs run at 1–2 Hz with 500–1500 ms latency; robot joint control requires **50 Hz (20 ms)**.
2. VLMs lack continuous physical inductive biases; asking an LLM to generate raw joint angles produces jerky, catastrophic collisions.
3. The **trainable PyTorch policy (ACT)** learns the actual high-dimensional, millimeter-accurate continuous trajectory manifold from camera pixels to motor torques.

---

## 2. The Dataset Specification

In standard supervised learning, data is represented as independent $(x, y)$ pairs. In imitation learning, data is structured as sequential Markov Decision Process trajectories called **Episodes**.

### 2.1 What is an Episode?
An episode is one complete execution of a manipulation task from initial reset to goal completion or termination:
$$\tau = \left\{ (o_0, a_0), (o_1, a_1), (o_2, a_2), \dots, (o_{T-1}, a_{T-1}) \right\}$$

- **Horizon $T$**: Typically $150\text{--}250$ timesteps @ 50 Hz ($3.0\text{--}5.0$ seconds of physical simulation).
- **Sampling Frequency**: $50\text{ Hz}$ ($\Delta t = 20\text{ ms}$).

### 2.2 Observation & Action Space Tensors

At each timestep $t$, the state observation $o_t$ and action $a_t$ consist of:

| Feature Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `observation.images.top` | `(480, 640, 3)` | `uint8` | Overhead perspective camera covering table workspace |
| `observation.images.wrist` | `(480, 640, 3)` | `uint8` | Gripper eye-in-hand camera providing fine-grained alignment |
| `observation.state` | `(7,)` | `float32` | Current 6 joint angles $[q_1, \dots, q_6] \in [-\pi, \pi]$ + gripper width $q_7$ |
| `observation.environment_state` | `(13,)` | `float32` | Goal vector: $[\mathbf{p}_{\text{target}}^{3D}, \mathbf{p}_{\text{dest}}^{3D}, \mathbf{e}_{\text{subgoal}}]$ (`FeatureType.ENV`), with $\mathbf{e}_{\text{subgoal}} \in \mathbb{R}^7$ |
| `action` | `(7,)` | `float32` | Target joint positions for step $t+1$ sent to MuJoCo actuators |
| `task_index` | `()` | `int64` | Discrete task identifier |

### 2.3 How Big is the Dataset?

Unlike computer vision classification datasets that require millions of images, Action Chunking with Transformers (ACT) is designed for **extreme sample efficiency**:

* **Academic Baseline (Zhao et al., Stanford RSS 2023)**: ACT achieves $>90\%$ task success with **50 demonstrations**.
* **Prototype / Verification Set**: $10\text{--}25$ episodes ($\approx 2{,}500\text{--}5{,}000$ transitions, $\sim 150\text{ MB}$).
* **Full Benchmark Set**: $50\text{--}100$ episodes ($\approx 10{,}000\text{--}25{,}000$ transitions, $\sim 1\text{--}2\text{ GB}$).

Because demonstrations are generated algorithmically using our MuJoCo expert solver running at $\sim 500\text{ FPS}$, recording 50 complete episodes takes **less than 2 minutes**.

### 2.4 Storage Format: Hugging Face `LeRobotDataset` v3.0
Data is not stored as loose uncompressed PNG files (which degrades disk I/O and training throughput). Instead, it uses the official Hugging Face LeRobot v3.0 standard:
1. **Metadata & Tensors**: Apache Parquet files (`data/chunk-000/episode_000000.parquet`) storing numerical vectors and frame indices.
2. **Video Streams**: Chunked MP4 files encoded with H.264 (`videos/chunk-000/observation.images.top.mp4`). PyTorch reads these via `torchvision` / PyAV hardware-accelerated decoders.
3. **Normalization Statistics (`meta/stats.json`)**: Empirical mean and standard deviation $(\mu_s, \sigma_s)$ and $(\mu_a, \sigma_a)$ computed over the entire dataset for z-score normalization:
   $$\tilde{s}_t = \frac{s_t - \mu_s}{\sigma_s + \epsilon}, \quad \tilde{a}_t = \frac{a_t - \mu_a}{\sigma_a + \epsilon}$$
4. **Finalization Lifecycle**: `dataset.finalize()` is explicitly invoked after all episodes are recorded to build the global index and compute dataset-wide statistics.

---

## 3. Model Architecture: Action Chunking with CVAE (ACT)

The core trainable architecture is **Action Chunking with Transformers (ACT)** (Zhao et al., RSS 2023).

```
Demonstration Action Chunk A_t ∈ ℝ^(50×7)    Proprioception State s_t ∈ ℝ^7
                  │                                        │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   CVAE Encoder (BERT-like)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼ Latent distribution q_ϕ(z | A_t, s_t)
                             z ~ 𝒩(μ_z, Σ_z) ∈ ℝ^32
                                      │
                                      ▼
Top RGB Camera  Wrist RGB Camera   Latent z   Proprioception s_t
     │                  │              │               │
     ▼                  ▼              │               │
┌─────────┐        ┌─────────┐         │               │
│ResNet-18│        │ResNet-18│         │               │
└────┬────┘        └────┬────┘         │               │
     │                  │              │               │
     └─────────┬────────┘              │               │
               ▼                       ▼               ▼
     ┌────────────────────────────────────────────────────────┐
     │           Transformer Encoder (Self-Attention)         │
     └─────────────────────────┬──────────────────────────────┘
                               │ Context Tokens
                               ▼
     ┌────────────────────────────────────────────────────────┐
     │           Transformer Decoder (Cross-Attention)        │
     │            Query Tokens Q ∈ ℝ^(50×d_model)             │
     └─────────────────────────┬──────────────────────────────┘
                               │
                               ▼
             Predicted Action Chunk Â_t ∈ ℝ^(50×7)
             [â_t, â_{t+1}, â_{t+2}, ..., â_{t+49}]
```

### 3.1 Why Predict Action "Chunks" Instead of Single Steps?
Single-step behavioral cloning ($\pi(a_t \mid s_t)$) fails in physical manipulation due to **compounding temporal error**:
$$\text{Drift} \propto \mathcal{O}(T^2)$$
If the policy deviates by $1\text{ mm}$ at step 1, step 2 receives an out-of-distribution observation, drifting further until the arm flails.

ACT predicts a chunk of future actions:
$$\mathbf{A}_t = [a_t, a_{t+1}, \dots, a_{t+K-1}] \in \mathbb{R}^{K \times d_a} \quad (K=50, d_a=7)$$
This gives the policy an implicit trajectory plan over a $1.0\text{-second}$ horizon.

### 3.2 Why a CVAE? (Solving the Multimodal Mode-Collapse Problem)
Human or algorithmic demonstrations are inherently multimodal:
- To pick up a cube, you can reach from the left or from the right.
- If a standard MSE loss is trained on multimodal data:
  $$\hat{a} = \frac{a_{\text{left}} + a_{\text{right}}}{2} \implies \text{Straight into the obstacle!}$$
The Conditional Variational Autoencoder (CVAE) introduces a continuous latent style variable $z \sim \mathcal{N}(0, I)$ that captures trajectory style/mode.

### 3.3 Loss Function
The model is trained end-to-end minimizing the CVAE variational bound:
$$\mathcal{L}_{\text{ACT}}(\theta, \phi) = \mathbb{E}_{z \sim q_\phi(z \mid \mathbf{A}_t, s_t)} \left[ \frac{1}{K} \sum_{k=0}^{K-1} \| a_{t+k} - \hat{a}_{t+k}(\theta, z) \|_1 \right] + \beta D_{\text{KL}}\left( q_\phi(z \mid \mathbf{A}_t, s_t) \,\parallel\, \mathcal{N}(0, I) \right)$$

Where:
1. **Reconstruction Loss**: L1 norm between ground-truth action chunk and predicted chunk. L1 is preferred over L2 because it penalizes trajectory deviations linearly, preserving sharp contact events.
2. **KL Divergence**: Regularizes the encoder posterior $q_\phi(z \mid \mathbf{A}, s)$ towards standard Gaussian prior $\mathcal{N}(0, I)$.
3. **$\beta$ hyperparameter**: Set to $\beta = 10.0$ to balance reconstruction precision and latent smoothness.

### 3.4 Inference Execution Modes & Action Queuing
* **Primary Benchmark Mode (Queue / Receding Horizon)**: `chunk_size = 50`, `n_action_steps = 10`, `temporal_ensemble_coeff = None`. ACT predicts a 50-step action chunk (1.0 s horizon). The 50 Hz control loop consumes $n_{\text{action\_steps}} = 10$ actions before triggering the next inference (~5 Hz cadence). Upon disturbance recovery, `policy.reset()` flushes the action queue immediately.
* **Secondary Ablation Mode (Temporal Action Ensembling / EMA)**: The policy queries a new chunk every timestep and computes an **Exponential Moving Average (EMA)** across overlapping predictions:
  $$a_t = \frac{\sum_{i=0}^{\min(t, K-1)} w_i \cdot \mathbf{A}_{t-i}[i]}{\sum_{i=0}^{\min(t, K-1)} w_i}, \quad w_i = \exp(-m \cdot i)$$
  Where $m = 0.01\text{--}0.05$ is the decay weighting.

---

## 4. Hardware Budget & Hyperparameters (NVIDIA RTX 3070 8GB)

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Vision Backbone** | 2x ResNet-18 (Pretrained) | Compact, fast feature extraction (~11M params each) |
| **Transformer Hidden Dim ($d_{\text{model}}$)** | $512$ | Balances representational capacity and GPU memory |
| **Transformer Layers** | 4 Encoder / 7 Decoder | 8 attention heads per layer |
| **Latent Dimension ($\dim(z)$)** | $32$ | Sufficient capacity for manipulation trajectory styles |
| **Action Chunk Horizon ($K$)** | $50$ | Exactly 1.0 second of physical planning @ 50 Hz |
| **Batch Size** | $16$ (or $32$) | Fits comfortably within 8GB VRAM |
| **Optimizer** | AdamW | $\text{lr} = 1\times 10^{-4}$, weight decay $= 1\times 10^{-4}$ |
| **LR Schedule** | Cosine Annealing | 500-step linear warmup, min lr $= 1\times 10^{-6}$ |
| **Precision** | PyTorch AMP (BF16/FP16) | Fast Tensor Core execution |
| **Peak VRAM** | **Planning Envelope: ~3.5–5.5 GB** | **Sufficient headroom on RTX 3070 (8GB); validate in Gate 0** |
| **Training Time** | **Planning Envelope: ~45–60 min** | 50,000 steps on local RTX 3070; validate in Gate 0 |

---

## 5. Summary: What Happens Step-by-Step

1. **Step 1: Record Dataset (`scripts/record_dataset.py`)**
   The algorithmic expert executes 50 pick-and-place trials in MuJoCo, saving camera video and joint state into `data/lerobot_embodied_arm/`.
2. **Step 2: Train Policy (`scripts/train_policy.py`)**
   PyTorch trains the ACT CVAE transformer on the local GPU, saving weights to `outputs/checkpoints/act_embodied_arm/`.
3. **Step 3: Closed-Loop Agentic Rollout (`scripts/evaluate.py`)**
   - Gemini decomposes human commands and locates objects.
   - The trained ACT policy reads real-time camera pixels and outputs action chunks.
   - MuJoCo simulates the contact dynamics.
   - If the cube slips, Gemini catches it and commands an autonomous retry.
