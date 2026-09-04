# Visuomotor Diffusion Policy: Algorithmic Foundations, Score Matching & Architectural Variants

> **Foundational Paper:**  
> *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*  
> Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song  
> *(Columbia University & Toyota Research Institute)*  
> *Robotics: Science and Systems (RSS), 2023 / International Journal of Robotics Research (IJRR), 2024* — [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Challenge of Multimodal Continuous Action Distributions
In visuomotor robot learning, demonstration datasets collected via human teleoperation are inherently **multimodal**, **non-Gaussian**, and **high-dimensional**:
* **Spatial Multimodality**: When maneuvering around an obstacle, moving to the left or right are both optimal modes; averaging them results in a straight-line collision into the obstacle.
* **Temporal / Velocity Multimodality**: An operator may hesitate, accelerate, or pause during a precision insertion task.

Traditional policy representations exhibit critical limitations when modeling multimodal action spaces:

```
+--------------------------------------------------------------------------------------------------+
| COMPARISON OF POLICY REPRESENTATIONS FOR MULTIMODAL CONTINUOUS CONTROL                           |
+----------------------+-----------------------------+---------------------------------------------+
| Formulation          | Mathematical Representation | Fundamental Limitation in Manipulation     |
+----------------------+-----------------------------+---------------------------------------------+
| Gaussian Regression  | a ~ N(mu_theta(s), Sigma)   | Mode collapse; averages distinct modes into |
|                      |                             | unfeasible intermediate trajectories.       |
+----------------------+-----------------------------+---------------------------------------------+
| Gaussian Mixture     | a ~ sum_i w_i N(mu_i, S_i)  | Numerical instability in EM/backprop; fixed |
| Models (GMM)         |                             | mode count hyperparameter k; mode collapse. |
+----------------------+-----------------------------+---------------------------------------------+
| Energy-Based Models  | p(a|s) proportional to      | Intractable partition function Z(s); MCMC   |
| (EBM / Implicit)     | exp(-E_theta(s, a))         | sampling during training is unstable.       |
+----------------------+-----------------------------+---------------------------------------------+
| Conditional VAE      | a = g_theta(s, z),          | Information bottleneck causes posterior     |
| (ACT style)          | z ~ N(0, I)                 | collapse or blur across close modes.        |
+----------------------+-----------------------------+---------------------------------------------+
| Diffusion Policy     | p(a|s) modeled via learned  | Expresses arbitrary continuous distributions|
| (DDPM / DDIM)        | score function / denoising  | with stable L2 training and no mode blur.   |
+----------------------+-----------------------------+---------------------------------------------+
```

### 1.2 Receding Horizon Trajectory Formulation
Diffusion Policy formulates policy execution as a **Receding Horizon Control (RHC)** problem over three distinct temporal horizons:

1. **Observation Horizon ($T_{\text{obs}}$)**: The policy conditions on a short history of recent observations (e.g., $T_{\text{obs}} = 2$ frames: $t-1$ and $t$) to capture object velocities and contact dynamics:
   $$\mathbf{O}_t = [o_{t - T_{\text{obs}} + 1}, \dots, o_t]$$
2. **Prediction Horizon ($T_p$)**: The policy denoises and predicts an action sequence spanning $T_p$ future steps (typically $T_p = 16$ steps):
   $$\mathbf{A}_t^{(0)} = [a_t, a_{t+1}, \dots, a_{t + T_p - 1}] \in \mathbb{R}^{T_p \times d_a}$$
3. **Execution Horizon ($T_a$)**: The controller executes only the first $T_a \le T_p$ steps (typically $T_a = 8$ steps) open-loop on the physical robot before triggering another diffusion cycle at timestep $t + T_a$:
   $$\mathbf{A}_{\text{exec}} = [a_t, \dots, a_{t + T_a - 1}]$$

This receding horizon formulation guarantees continuous feedback and dynamic error recovery while avoiding single-step myopic drifting.

---

## 2. Mathematical Formulation: DDPM in Action Space

Visuomotor Diffusion Policy adapts **Denoising Diffusion Probabilistic Models (DDPM)** to model the conditional distribution $p(\mathbf{A}^{(0)} \mid \mathbf{O}_t)$.

```
FORWARD DIFFUSION PROCESS (Markovian Noise Injection, k = 0 -> N):
  A^(0) -----------> A^(1) -----------> ... -----------> A^(N) ~ N(0, I)
  (Ground Truth)      (+ beta_1)                           (Pure Gaussian Noise)

REVERSE DENOISING PROCESS (Score Matching Model epsilon_theta, k = N -> 0):
  A^(N) -----------> A^(N-1) ----------> ... -----------> A^(0)
  ~ N(0, I)           Conditioned on Observation O_t      (Recovered Trajectory)
```

### 2.1 The Forward (Noising) Diffusion Process
Given a ground-truth action trajectory $\mathbf{A}^{(0)} \sim q(\mathbf{A}^{(0)} \mid \mathbf{O}_t)$, the forward process is a discrete-time Markov chain that iteratively injects Gaussian noise across $N$ steps according to a variance schedule $\beta_1, \beta_2, \dots, \beta_N \in (0, 1)$:

$$q(\mathbf{A}^{(k)} \mid \mathbf{A}^{(k-1)}) = \mathcal{N}\left( \mathbf{A}^{(k)};\, \sqrt{1 - \beta_k} \, \mathbf{A}^{(k-1)},\, \beta_k \mathbf{I} \right)$$

Let $\alpha_k = 1 - \beta_k$ and define the cumulative product:

$$\bar{\alpha}_k = \prod_{s=1}^k \alpha_s = \prod_{s=1}^k (1 - \beta_s)$$

By recursive expansion, the marginal distribution of the noisy trajectory $\mathbf{A}^{(k)}$ at arbitrary diffusion step $k \in \{1, \dots, N\}$ can be evaluated in closed form without simulating intermediate steps:

$$\mathbf{A}^{(k)} = \sqrt{\alpha_k} \mathbf{A}^{(k-1)} + \sqrt{1 - \alpha_k} \boldsymbol{\epsilon}_{k-1}$$

$$= \sqrt{\alpha_k} \left( \sqrt{\alpha_{k-1}} \mathbf{A}^{(k-2)} + \sqrt{1 - \alpha_{k-1}} \boldsymbol{\epsilon}_{k-2} \right) + \sqrt{1 - \alpha_k} \boldsymbol{\epsilon}_{k-1}$$

$$= \sqrt{\alpha_k \alpha_{k-1}} \mathbf{A}^{(k-2)} + \sqrt{\alpha_k(1 - \alpha_{k-1})} \boldsymbol{\epsilon}_{k-2} + \sqrt{1 - \alpha_k} \boldsymbol{\epsilon}_{k-1}$$

Because the sum of two independent Gaussians $\mathcal{N}(\mathbf{0}, \sigma_1^2 \mathbf{I})$ and $\mathcal{N}(\mathbf{0}, \sigma_2^2 \mathbf{I})$ is distributed as $\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$:

$$\sqrt{\alpha_k(1 - \alpha_{k-1})}^2 + \sqrt{1 - \alpha_k}^2 = \alpha_k - \alpha_k \alpha_{k-1} + 1 - \alpha_k = 1 - \alpha_k \alpha_{k-1}$$

Induction yields the direct analytical sampling formula:

$$\mathbf{A}^{(k)} = \sqrt{\bar{\alpha}_k} \mathbf{A}^{(0)} + \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

$$q(\mathbf{A}^{(k)} \mid \mathbf{A}^{(0)}) = \mathcal{N}\left( \mathbf{A}^{(k)};\, \sqrt{\bar{\alpha}_k} \mathbf{A}^{(0)},\, (1 - \bar{\alpha}_k) \mathbf{I} \right)$$

### 2.2 The Reverse (Denoising) Process & Score Matching
The true reverse conditional distribution $q(\mathbf{A}^{(k-1)} \mid \mathbf{A}^{(k)}, \mathbf{A}^{(0)})$ is tractable via Bayes' rule:

$$q(\mathbf{A}^{(k-1)} \mid \mathbf{A}^{(k)}, \mathbf{A}^{(0)}) = \mathcal{N}\left( \mathbf{A}^{(k-1)};\, \tilde{\boldsymbol{\mu}}_k(\mathbf{A}^{(k)}, \mathbf{A}^{(0)}),\, \tilde{\beta}_k \mathbf{I} \right)$$

where the posterior mean $\tilde{\boldsymbol{\mu}}_k$ and variance $\tilde{\beta}_k$ are:

$$\tilde{\beta}_k = \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k} \beta_k$$

$$\tilde{\boldsymbol{\mu}}_k(\mathbf{A}^{(k)}, \mathbf{A}^{(0)}) = \frac{\sqrt{\bar{\alpha}_{k-1}} \beta_k}{1 - \bar{\alpha}_k} \mathbf{A}^{(0)} + \frac{\sqrt{\alpha_k}(1 - \bar{\alpha}_{k-1})}{1 - \bar{\alpha}_k} \mathbf{A}^{(k)}$$

Substituting the expression for $\mathbf{A}^{(0)}$ derived from $\mathbf{A}^{(k)} = \sqrt{\bar{\alpha}_k}\mathbf{A}^{(0)} + \sqrt{1 - \bar{\alpha}_k}\boldsymbol{\epsilon}$:

$$\mathbf{A}^{(0)} = \frac{\mathbf{A}^{(k)} - \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_k}}$$

reparameterizes the posterior mean solely as a function of $\mathbf{A}^{(k)}$ and the noise $\boldsymbol{\epsilon}$:

$$\tilde{\boldsymbol{\mu}}_k = \frac{1}{\sqrt{\alpha_k}} \left( \mathbf{A}^{(k)} - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}} \boldsymbol{\epsilon} \right)$$

In policy learning, we parameterize a deep neural network $\boldsymbol{\epsilon}_\theta(\mathbf{A}^{(k)}, k, \mathbf{O}_t)$ to estimate this injected noise $\boldsymbol{\epsilon}$.

### 2.3 Connection to Tweedie's Formula and Score-Based Generative Modeling
By Tweedie's formula, the posterior expectation of the clean variable given a Gaussian corrupted observation satisfies:

$$\mathbb{E}[\mathbf{A}^{(0)} \mid \mathbf{A}^{(k)}] = \frac{\mathbf{A}^{(k)}}{\sqrt{\bar{\alpha}_k}} + \frac{1 - \bar{\alpha}_k}{\sqrt{\bar{\alpha}_k}} \nabla_{\mathbf{A}^{(k)}} \log q(\mathbf{A}^{(k)})$$

Equating this to our reparameterization reveals the exact relationship between the noise prediction network $\boldsymbol{\epsilon}_\theta$ and the **score function** (the gradient of the log-density of the noisy action distribution):

$$\nabla_{\mathbf{A}^{(k)}} \log q(\mathbf{A}^{(k)} \mid \mathbf{O}_t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{A}^{(k)}, k, \mathbf{O}_t)}{\sqrt{1 - \bar{\alpha}_k}}$$

Thus, training the network to predict noise $\boldsymbol{\epsilon}$ is mathematically equivalent to **Denoising Score Matching**, learning the vector field that directs arbitrary noisy action points toward high-probability manifolds of expert demonstrations.

### 2.4 Training Loss Formulation
The network parameters $\theta$ are optimized using the simplified variational bound (L2 Mean Squared Error):

$$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{k \sim \mathcal{U}(1, N),\, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),\, (\mathbf{O}_t, \mathbf{A}^{(0)}) \sim \mathcal{D}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left( \sqrt{\bar{\alpha}_k} \mathbf{A}^{(0)} + \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon},\, k,\, \mathbf{O}_t \right) \right\|_2^2 \right]$$

---

## 3. Fast Inference Acceleration: DDPM vs DDIM

Standard DDPM requires $N = 100$ sequential neural evaluations during inference, which induces prohibitive latency ($\sim 120\,\text{ms}$) on robot hardware operating at $50\,\text{Hz}$.

### 3.1 Denoising Diffusion Implicit Models (DDIM) Formulation
**DDIM** (Song et al., 2020) generalizes DDPM to a class of **non-Markovian forward processes** that share the exact same marginal distributions $q(\mathbf{A}^{(k)} \mid \mathbf{A}^{(0)})$, allowing the same trained model $\boldsymbol{\epsilon}_\theta$ to be sampled without retraining.

Given a sub-sequence of $S$ sampling steps $\{\tau_1, \tau_2, \dots, \tau_S\} \subset \{1, \dots, N\}$ (where $S \ll N$, e.g., $S = 10$):

$$\mathbf{A}^{(\tau_{i-1})} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \underbrace{\left( \frac{\mathbf{A}^{(\tau_i)} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \boldsymbol{\epsilon}_\theta(\mathbf{A}^{(\tau_i)}, \tau_i, \mathbf{O}_t)}{\sqrt{\bar{\alpha}_{\tau_i}}} \right)}_{\text{Estimated } \mathbf{A}^{(0)}} + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2} \boldsymbol{\epsilon}_\theta(\mathbf{A}^{(\tau_i)}, \tau_i, \mathbf{O}_t) + \sigma_{\tau_i} \boldsymbol{\epsilon}$$

where $\sigma_{\tau_i}$ controls the stochasticity of the sampling trajectory:

$$\sigma_{\tau_i}(\eta) = \eta \sqrt{\frac{1 - \bar{\alpha}_{\tau_{i-1}}}{1 - \bar{\alpha}_{\tau_i}}} \sqrt{1 - \frac{\bar{\alpha}_{\tau_i}}{\bar{\alpha}_{\tau_{i-1}}}}$$

* When $\eta = 1$: The sampler matches the stochastic DDPM trajectory.
* When $\eta = 0$: The stochastic term vanishes entirely ($\sigma_{\tau_i} = 0$), yielding a **deterministic ODE trajectory** (the Probability Flow ODE).

```
SAMPLING EFFICIENCY & LATENCY TRADEOFF (RTX 4090 / RTX 3070):
+--------------------+------------+------------------+---------------------+
| Sampler Type       | Steps (S)  | GPU Latency (ms) | Control Rate (Hz)   |
+--------------------+------------+------------------+---------------------+
| Standard DDPM      | 100        | 115 ms           | ~8.7 Hz (Slow)      |
| DDPM Reduced       | 50         | 58 ms            | ~17.2 Hz            |
| DDIM Deterministic | 16         | 18 ms            | ~55 Hz (Real-time!) |
| DDIM Fast          | 8          | 9.5 ms           | ~105 Hz             |
+--------------------+------------+------------------+---------------------+
```

---

## 4. Architectural Variants: CNN vs Time-Conditioned Transformer

Diffusion Policy is commonly realized in two distinct architectural configurations:

```
+--------------------------------------------------------------------------------------------------+
| ARCHITECTURAL PARADIGMS                                                                          |
+--------------------------------------------------------------------------------------------------+
| 1. CNN-BASED 1D TEMPORAL U-NET                                                                   |
|                                                                                                  |
|   Noisy Chunk A^(k) in R^{T_p x d_a}                                                             |
|           |                                                                                      |
|           v                                                                                      |
|     [ Conv1D Down 1 ] ---- Skip Connection 1 ----> [ Conv1D Up 1 ]                              |
|           |                                                ^                                     |
|     [ Conv1D Down 2 ] ---- Skip Connection 2 ----> [ Conv1D Up 2 ]                              |
|           |                                                ^                                     |
|           +--------> [ Residual Mid Blocks ] --------------+                                     |
|                             ^                                                                    |
|                             | FiLM Modulation: gamma(e_cond) * h + beta(e_cond)                  |
|                      Observation e_obs + Diffusion Step e_k                                      |
|                                                                                                  |
| 2. TIME-CONDITIONED DIFFUSION TRANSFORMER (DiT)                                                  |
|                                                                                                  |
|   Action Tokens [a_0, ..., a_{T_p-1}] + Pos Embeddings                                          |
|           |                                                                                      |
|           v                                                                                      |
|   +-------------------------------------------------------------+                                |
|   | DiT Block: Adaptive LayerNorm (AdaLN)                       |                                |
|   | - Self-Attention across Trajectory Timesteps                |                                |
|   | - Cross-Attention to Observation Memory Sequence [O_t]      |                                |
|   | - Pointwise MLP with Time Embedding Injection               |                                |
|   +-------------------------------------------------------------+ (x N layers)                   |
|           |                                                                                      |
|           v                                                                                      |
|   Predicted Noise Chunk epsilon_theta in R^{T_p x d_a}                                           |
+--------------------------------------------------------------------------------------------------+
```

### 4.1 The CNN-Based 1D Temporal U-Net
The CNN backbone treats the trajectory $\mathbf{A}^{(k)} \in \mathbb{R}^{T_p \times d_a}$ as a 1D spatial signal with $d_a$ channels.
* **Temporal Convolutions**: $1 \times 5$ or $1 \times 3$ convolutional kernels operate over the time horizon $T_p$.
* **Conditioning via FiLM (Feature-wise Linear Modulation)**:  
  Given an encoded observation vector $\mathbf{e}_{\text{obs}} = \text{Encoder}(\mathbf{O}_t)$ and sinusoidal time step embedding $\mathbf{e}_k = \text{Embed}(k)$, a conditioning vector $\mathbf{c} = [\mathbf{e}_{\text{obs}} \,\|\, \mathbf{e}_k]$ modulates intermediate activation feature maps $h$:
  $$\operatorname{FiLM}(h;\, \mathbf{c}) = \gamma(\mathbf{c}) \odot h + \beta(\mathbf{c})$$
  where $\gamma(\cdot)$ and $\beta(\cdot)$ are learned linear projections outputting scale and shift vectors.

### 4.2 The Time-Conditioned Diffusion Transformer (DiT)
The transformer variant flattens each action timestep into a token vector $x_i \in \mathbb{R}^{d_{\text{model}}}$.
* **Adaptive LayerNorm (AdaLN)**: Rather than adding time embeddings, AdaLN regresses dimension-wise scaling and shifting parameters directly from the time embedding:
  $$\operatorname{AdaLN}(h;\, \mathbf{e}_k) = \gamma(\mathbf{e}_k) \odot \left( \frac{h - \mu}{\sigma} \right) + \beta(\mathbf{e}_k)$$
* **Cross-Attention Conditioning**: Visual observation tokens extracted from ResNet / ViT backbones are preserved as spatial tokens, enabling the trajectory tokens to cross-attend directly to specific physical regions in the scene.

### 4.3 Detailed Architectural Comparison

| Dimension | 1D Temporal U-Net | Time-Conditioned Transformer (DiT) |
| :--- | :--- | :--- |
| **Inductive Bias** | Strong local temporal translation equivariance | Global attention across all trajectory tokens |
| **Parameter Count** | Compact ($\sim 15\text{M} - 35\text{M}$ parameters) | Medium-to-Large ($\sim 50\text{M} - 120\text{M}$ parameters) |
| **Compute / VRAM** | Very low ($\approx 1.8\,\text{GB}$ inference VRAM) | Moderate ($\approx 3.2\,\text{GB}$ inference VRAM) |
| **Horizon Flexibility** | Requires fixed $T_p$ matching kernel strides | Supports variable sequence lengths $T_p$ |
| **Best Suited For** | High-speed low-latency industrial assembly | Rich semantic tasks with multimodal prompt conditioning |

---

## 5. Algorithmic Pseudocode (PyTorch / LeRobot Standard)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionPolicyLoss(nn.Module):
    """
    Computes noise prediction MSE loss for DDPM Diffusion Policy.
    """
    def __init__(self, model: nn.Module, num_diffusion_steps: int = 100):
        super().__init__()
        self.model = model
        self.num_steps = num_diffusion_steps

        # Cosine beta schedule (Nichol & Dhariwal, 2021)
        betas = self._cosine_beta_schedule(num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, n_steps: int, s: float = 0.008) -> torch.Tensor:
        steps = n_steps + 1
        x = torch.linspace(0, n_steps, steps)
        alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward(self, obs_cond: torch.Tensor, action_trajectory: torch.Tensor) -> torch.Tensor:
        """
        obs_cond: (B, obs_dim) or (B, N_tokens, d_model)
        action_trajectory: (B, T_p, action_dim) ground-truth action chunk
        """
        batch_size = action_trajectory.shape[0]
        device = action_trajectory.device

        # 1. Sample uniform random diffusion timestep k in [0, num_steps - 1]
        k = torch.randint(0, self.num_steps, (batch_size,), device=device).long()

        # 2. Sample standard normal Gaussian noise
        noise = torch.randn_like(action_trajectory)

        # 3. Compute noisy trajectory A^(k) analytically
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[k].view(batch_size, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[k].view(batch_size, 1, 1)
        noisy_action = sqrt_alpha_bar * action_trajectory + sqrt_one_minus_alpha_bar * noise

        # 4. Predict noise with policy network
        pred_noise = self.model(noisy_action, k, obs_cond)

        # 5. Mean squared error objective
        loss = F.mse_loss(pred_noise, noise)
        return loss
```

---

## 6. Mathematical Summary & Key Takeaways

1. **Exact Score Matching**: Visuomotor diffusion trains a neural network $\boldsymbol{\epsilon}_\theta$ to model the normalized gradient of the trajectory probability distribution $\nabla_{\mathbf{A}} \log p(\mathbf{A} \mid \mathbf{O}_t)$.
2. **Elimination of Posterior Collapse**: Unlike CVAEs, there is no adversarial regularization term ($\beta D_{\text{KL}}$) competing against reconstruction. The pure L2 noise prediction loss trains stably without tuning delicate balance hyperparameters.
3. **Decoupled Observation & Action Horizons**: The system cleanly operates over an observation window $T_{\text{obs}} = 2$, prediction window $T_p = 16$, and execution window $T_a = 8$, providing a resilient balance of reactivity and long-term planning.
4. **Deterministic Fast Sampling (DDIM)**: By stepping down the probability flow ODE in $10\text{--}16$ steps, inference executes in under $20\,\text{ms}$, satisfying real-time $50\,\text{Hz}$ deployment constraints on standard NVIDIA GPUs.
