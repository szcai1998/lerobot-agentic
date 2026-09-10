# Embodied AI Algorithmic Foundations: Mathematical Compendium

> **Role:** Background reference only. This suite documents the math behind the
> robot-policy families that **LeRobot Reliability Lab** wraps as frozen,
> off-the-shelf policies — it is *not* an implementation spec and the project does
> not train these architectures. For the engineering system see
> [`../LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md`](../LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md)
> (§7 Policy Abstraction, §8 Compute) and the landscape snapshot
> [`../FRONTIER_RESEARCH_SUPPORT.md`](../FRONTIER_RESEARCH_SUPPORT.md).
>
> **Policy roster covered:** ACT (historical/local baseline), Diffusion Policy
> (context), SmolVLA (mandatory modifiable VLA), $\pi_{0.5}$ / MolmoAct2 (strong
> frozen reference). Reference frameworks: `lerobot==0.6.1`, `mujoco>=3.12`,
> PyTorch 2.6+ (CUDA 12.4+).

---

## 1. Compendium Index

| Chapter | Document | Foundational Literature | Core Focus & Equations |
| :---: | :--- | :--- | :--- |
| **01** | [`01_act_policy.md`](./01_act_policy.md) | Zhao et al. (RSS 2023) | Action Chunking with Transformers (ACT), CVAE ELBO derivation, Gaussian KL divergence, temporal ensembling EMA. |
| **02** | [`02_diffusion_policy.md`](./02_diffusion_policy.md) | Chi et al. (RSS 2023 / IJRR 2024) | Visuomotor Diffusion Policy, DDPM forward/reverse equations, Tweedie's score matching, DDIM ODE acceleration, CNN vs DiT. |
| **03** | [`03_vla_flow_matching.md`](./03_vla_flow_matching.md) | LeRobot Team (2025), Black et al. (2024/2025), Lipman et al. (ICLR 2023) | SmolVLA-450M, $\pi_0$ & $\pi_0$-FAST, Continuous Optimal Transport Flow Matching (OT-CFM) vs Diffusion, DCT frequency tokenization. |

---

## 2. Cross-Comparison of Policy Families

Indicative characteristics of the architectures in the frozen policy roster. VRAM
figures are order-of-magnitude references from the source literature, **not**
measured on this project's hardware — see the Technical Report §8 for the
4090-first measurement contract.

| Paradigm | Architecture | Params | Control Rate | Inductive Bias | Multimodality | Sample Efficiency | Role in this project |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ACT** *(Zhao et al.)* | ResNet-18 / ViT + Transformer CVAE | ~45M–85M | ~50 Hz | Contiguous temporal trajectory coherence | CVAE latent $z \sim \mathcal{N}(0, I)$ | Very high (~50 demos) | Historical/local sanity baseline; deterministic queue semantics for debugging recovery infra |
| **Diffusion Policy (CNN / DiT)** *(Chi et al.)* | 1D temporal U-Net / time-conditioned DiT | ~15M–120M | ~30–60 Hz (DDIM-16) | Temporal translation equivariance; global cross-attention (DiT) | Score-based stochastic gradient field | Moderate–high (~100–150 demos) | Context only — not in the mandatory policy roster |
| **SmolVLA-450M** *(Hugging Face)* | SigLIP + SmolLM + flow-matching head | 450M | ~50 Hz (Heun-2) | Decoupled semantic reasoning + continuous flow | Optimal-transport straight vector fields | High (pretrained) | **Mandatory modifiable VLA**: language conditioning, local inference, lightweight adaptation, detector/recovery integration |
| **$\pi_0$ / $\pi_{0.5}$ / $\pi_0$-FAST** *(Physical Intelligence)* | PaliGemma-3B + continuous flow / DCT | ~3.2B | ~50 Hz (FAST/1-step) | Frequency-space energy compaction (DCT) | Continuous CFM / autoregressive discrete tokens | Extremely high (pretrained) | **Strong frozen reference** (one-of with MolmoAct2, chosen by measured 4090 feasibility); cross-policy test target |

For semantic high-level recovery the Technical Report (§12.2) allows a cloud
multimodal reasoner such as `gemini-robotics-er-2` as *one* implementation of the
`RecoveryExpert` interface; it is a plug-in, not a policy in the roster.

---

## 3. Unified Mathematical Notation Dictionary

Across all documents in `docs/algorithm/`, symbols conform to this notation.

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
