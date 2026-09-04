# Action Chunking with Transformers (ACT): Algorithmic Formulation & Architectural Foundations

> **Foundational Paper:**  
> *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*  
> Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn (Stanford University)  
> *Robotics: Science and Systems (RSS), 2023* — [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Compounding Error Dilemma in Behavioral Cloning
Classical Imitation Learning via Behavioral Cloning (BC) formulates robotic control as a Markovian regression problem. At each discrete timestep $t$, an agent observes environmental state $s_t \in \mathcal{S}$ (e.g., camera RGB streams and robot proprioception) and predicts an instantaneous motor action $a_t \in \mathcal{A}$:

$$\pi_\theta: \mathcal{S} \to \mathcal{A}, \quad a_t = \pi_\theta(s_t)$$

Under the standard forward Kullback-Leibler (KL) divergence objective:

$$\min_\theta \mathbb{E}_{(s_t, a_t) \sim \mathcal{D}_{\text{expert}}} \left[ -\log \pi_\theta(a_t \mid s_t) \right]$$

In non-ergodic environments, this formulation suffers from severe **covariate shift** and **compounding execution errors**. If the learned policy deviates from the expert demonstration by a small perturbation $\epsilon$:

$$\mathbb{E}_{s \sim \mathcal{D}_{\text{expert}}} \left[ \|\pi^*(s) - \pi_\theta(s)\| \right] \le \epsilon$$

the cumulative trajectory error over an execution horizon $T$ scales quadratically:

$$\text{Error}_{\text{cumulative}} = \mathcal{O}(\epsilon \cdot T^2)$$

Intuitively, an unforced error at step $t$ moves the robot into an unseen state $s_{t+1} \notin \mathcal{D}_{\text{expert}}$, where the policy output is undefined or erratic, driving the system into catastrophic task failure.

### 1.2 Action Chunking as Error Compounding Mitigation
**Action Chunking with Transformers (ACT)** alters this formulation by framing prediction over an **extended temporal horizon**. Rather than predicting a single step $a_t$, the policy predicts an **action chunk** $\mathbf{A}_t$ containing $K$ consecutive future actions:

$$\mathbf{A}_t = \begin{bmatrix} a_t \\ a_{t+1} \\ \vdots \\ a_{t+K-1} \end{bmatrix} \in \mathbb{R}^{K \times d_a}$$

where $d_a$ is the robot's degree-of-freedom (DoF) action dimension (e.g., $d_a = 7$ for a single 6-DoF arm + 1-DoF parallel gripper, or $d_a = 14$ for a bimanual setup).

By shifting the effective decision horizon from $T$ single steps to $T/K$ chunks, the upper bound on the compounding trajectory error contracts from quadratic to linear in the effective number of decisions:

$$\text{Error}_{\text{chunked}} = \mathcal{O}\left( \epsilon \cdot \frac{T^2}{K} \right) \quad \xrightarrow{K \to T} \quad \mathcal{O}(\epsilon \cdot T)$$

Furthermore, predicting contiguous chunks enforces **temporal smoothness** and preserves kinematic consistency across fine-motor sub-trajectories (such as slotting an audio jack or threading a needle).

---

## 2. Generative Modeling via Conditional VAE (CVAE)

Human teleoperation demonstrations exhibit inherent **multimodality**: given identical initial scene observations $s_t$, a human operator may pick up an object from the left or right, or grasp different parts of a tool. Deterministic regression with Mean Squared Error (MSE) or L1 losses under multimodal targets converges to the **mode average** (the conditional expectation $\mathbb{E}[a \mid s]$), resulting in physically invalid collisions or indecisive drifting.

To capture multimodal distributions $p(\mathbf{A}_t \mid s_t)$, ACT models the joint distribution through a **Conditional Variational Autoencoder (CVAE)** parameterized by a latent style variable $z \in \mathbb{R}^{d_z}$ (typically $d_z = 32$).

```
                    +------------------------------------------+
                    |           Training Only (Encoder)        |
                    |                                          |
                    |  Chunk A_t in R^{K x d_a}                |
                    |  Current Joint Pos a_bar_t in R^{d_a}    |
                    +--------------------+---------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Transformer Encoder q_phi   |
                         +---------------+---------------+
                                         |
                                  mu_phi, sigma_phi
                                         |
                                         v   Reparameterization
                                 z ~ N(mu, sigma^2)
                                         |
                                         v
+------------------------+       +-------+-------+       +------------------------+
|  Camera RGB Images     | ----> |  Transformer  | <---- |  Current Joint State   |
|  (Top, Wrist, Overhead)|       |  Decoder pi_theta|    |  a_bar_t in R^{d_a}    |
+------------------------+       +-------+-------+       +------------------------+
                                         |
                                         v
                                Predicted Chunk
                             A_hat_t in R^{K x d_a}
```

### 2.1 The Evidence Lower Bound (ELBO) Derivation
The true conditional log-likelihood of the action trajectory $\mathbf{A}_t$ given the state $s_t$ satisfies:

$$\log p_\theta(\mathbf{A}_t \mid s_t) = \log \int p_\theta(\mathbf{A}_t, z \mid s_t) \, dz = \log \int q_\phi(z \mid \mathbf{A}_t, s_t) \frac{p_\theta(\mathbf{A}_t, z \mid s_t)}{q_\phi(z \mid \mathbf{A}_t, s_t)} \, dz$$

Applying Jensen's Inequality to the concave logarithm function:

$$\log p_\theta(\mathbf{A}_t \mid s_t) \ge \mathbb{E}_{z \sim q_\phi(z \mid \mathbf{A}_t, s_t)} \left[ \log \frac{p_\theta(\mathbf{A}_t, z \mid s_t)}{q_\phi(z \mid \mathbf{A}_t, s_t)} \right]$$

$$= \mathbb{E}_{z \sim q_\phi(z \mid \mathbf{A}_t, s_t)} \left[ \log \frac{p_\theta(\mathbf{A}_t \mid s_t, z) \, p(z \mid s_t)}{q_\phi(z \mid \mathbf{A}_t, s_t)} \right]$$

$$= \mathbb{E}_{z \sim q_\phi(z \mid \mathbf{A}_t, s_t)} \left[ \log p_\theta(\mathbf{A}_t \mid s_t, z) \right] - D_{\text{KL}}\left( q_\phi(z \mid \mathbf{A}_t, s_t) \,\parallel\, p(z \mid s_t) \right)$$

Assuming an uninformative state-independent prior $p(z \mid s_t) = p(z) = \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_z})$, the Variational Lower Bound (ELBO) is:

$$\mathcal{L}_{\text{ELBO}}(\theta, \phi) = \mathbb{E}_{z \sim q_\phi(z \mid \mathbf{A}_t, s_t)} \left[ \log p_\theta(\mathbf{A}_t \mid s_t, z) \right] - D_{\text{KL}}\left( q_\phi(z \mid \mathbf{A}_t, s_t) \,\parallel\, \mathcal{N}(\mathbf{0}, \mathbf{I}) \right)$$

### 2.2 Analytical Formulation of the Gaussian KL Divergence
The encoder parameterizes a diagonal Gaussian posterior:

$$q_\phi(z \mid \mathbf{A}_t, s_t) = \mathcal{N}\left( z;\, \boldsymbol{\mu}_\phi, \operatorname{diag}(\boldsymbol{\sigma}_\phi^2) \right)$$

where $\boldsymbol{\mu}_\phi, \log \boldsymbol{\sigma}_\phi^2 \in \mathbb{R}^{d_z}$. The Kullback-Leibler divergence between two multivariate Gaussians in $\mathbb{R}^{d_z}$:

$$p(z) = \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad q(z) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad \boldsymbol{\Sigma} = \operatorname{diag}(\sigma_1^2, \dots, \sigma_{d_z}^2)$$

is derived analytically as:

$$D_{\text{KL}}(q \parallel p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz$$

$$= \frac{1}{2} \left[ \operatorname{Tr}\left(\mathbf{I}^{-1} \boldsymbol{\Sigma}\right) + (\mathbf{0} - \boldsymbol{\mu})^T \mathbf{I}^{-1} (\mathbf{0} - \boldsymbol{\mu}) - d_z + \ln\left(\frac{\det \mathbf{I}}{\det \boldsymbol{\Sigma}}\right) \right]$$

$$= \frac{1}{2} \left[ \sum_{j=1}^{d_z} \sigma_j^2 + \sum_{j=1}^{d_z} \mu_j^2 - d_z - \sum_{j=1}^{d_z} \ln(\sigma_j^2) \right]$$

$$= -\frac{1}{2} \sum_{j=1}^{d_z} \left( 1 + \ln(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

### 2.3 Reparameterization Trick & Loss Objective
To compute gradients $\nabla_\phi$ through the stochastic sampling step, ACT applies the reparameterization trick:

$$z = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_z})$$

In ACT, the reconstruction term assumes a Laplace-distributed likelihood over action sequences, yielding an $\mathcal{L}_1$ reconstruction penalty rather than $\mathcal{L}_2$. The total loss minimized over dataset $\mathcal{D}$ is:

$$\mathcal{L}_{\text{ACT}}(\theta, \phi) = \mathbb{E}_{(\mathbf{A}_t, s_t) \sim \mathcal{D}} \left[ \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \sum_{k=0}^{K-1} \| a_{t+k} - \hat{a}_{t+k}(\theta, s_t, z) \|_1 \right] + \beta D_{\text{KL}}\left( q_\phi(z \mid \mathbf{A}_t, s_t) \,\parallel\, \mathcal{N}(\mathbf{0}, \mathbf{I}) \right) \right]$$

where $\beta > 0$ is a hyperparameter balancing latent capacity against trajectory precision (typically $\beta \in [10, 100]$ in practice to prevent posterior collapse while maintaining tight trajectory tracking).

---

## 3. Transformer Encoder-Decoder Architecture

ACT implements a custom transformer architecture consisting of:
1. **Vision Backbones**: Convolutional (ResNet-18) or Transformer (DINOv2 ViT-B / SigLIP) networks extracting visual feature tokens.
2. **CVAE Transformer Encoder**: Compresses the trajectory chunk $\mathbf{A}_t$ and current joint positions $\bar{a}_t$ into Gaussian parameters $[\boldsymbol{\mu}, \log \boldsymbol{\sigma}^2]$.
3. **Policy Transformer Decoder**: Synthesizes the predicted action chunk $\hat{\mathbf{A}}_t$ conditioned on visual tokens, joint state, and latent vector $z$.

```
+-----------------------------------------------------------------------------------------+
|                                    ACT POLICY DECODER                                   |
|                                                                                         |
|  Learned Positional Queries           Cross-Attention Memory Sequence                   |
|  [Q_0, Q_1, ..., Q_{K-1}]              [V_{cam1}, V_{cam2}, E_{state}, E_z]             |
|          |                                              |                               |
|          +----------------------+-----------------------+                               |
|                                 v                                                       |
|             +---------------------------------------+                                   |
|             |  Transformer Decoder Layers (x6)      |                                   |
|             |  - Self-Attention over Queries        |                                   |
|             |  - Cross-Attention to Visual & Latent |                                   |
|             |  - Feed-Forward MLP (d_ff = 2048)     |                                   |
|             +-------------------+-------------------+                                   |
|                                 |                                                       |
|                                 v                                                       |
|                     Linear Output Projection                                            |
|                                 |                                                       |
|                                 v                                                       |
|         A_hat_t = [a_t, a_{t+1}, ..., a_{t+K-1}] in R^{K x d_a}                         |
+-----------------------------------------------------------------------------------------+
```

### 3.1 Tokenization and Input Embeddings
Let the multi-camera observation at timestep $t$ comprise $M$ RGB viewpoints:

$$\mathbf{I}_t = \{ I_t^{(1)}, I_t^{(2)}, \dots, I_t^{(M)} \}, \quad I_t^{(m)} \in \mathbb{R}^{3 \times H \times W}$$

Each image is processed by a visual backbone (e.g., ResNet-18 with the classification head removed) to produce spatial feature maps:

$$F^{(m)} = \text{Backbone}(I_t^{(m)}) \in \mathbb{R}^{C \times h \times w}$$

The feature map is projected to model dimension $d_{\text{model}}$ via a $1 \times 1$ 2D convolution and flattened:

$$V^{(m)} = \text{Flatten}\left( \operatorname{Conv2D}_{1\times1}(F^{(m)}) \right) + \mathbf{P}_{\text{2D}} \in \mathbb{R}^{(hw) \times d_{\text{model}}}$$

where $\mathbf{P}_{\text{2D}} \in \mathbb{R}^{(hw) \times d_{\text{model}}}$ is a fixed sinusoidal 2D spatial positional embedding.

The robot proprioceptive joint state $\bar{a}_t \in \mathbb{R}^{d_a}$ and the sampled style latent $z \in \mathbb{R}^{d_z}$ are projected via learned linear maps:

$$E_{\text{state}} = \mathbf{W}_{\text{state}} \bar{a}_t + b_{\text{state}} \in \mathbb{R}^{1 \times d_{\text{model}}}$$

$$E_z = \mathbf{W}_z z + b_z \in \mathbb{R}^{1 \times d_{\text{model}}}$$

The full encoder memory sequence $\mathbf{M} \in \mathbb{R}^{(M \cdot hw + 2) \times d_{\text{model}}}$ fed into the decoder cross-attention blocks is concatenated as:

$$\mathbf{M} = \left[ V^{(1)} \,\|\, V^{(2)} \,\|\, \dots \,\|\, V^{(M)} \,\|\, E_{\text{state}} \,\|\, E_z \right]$$

### 3.2 Action Sequence Generation via Learned Positional Queries
The policy decoder does not autoregressively generate actions one-by-one. Instead, it predicts the entire chunk $\mathbf{A}_t$ **in parallel** (non-autoregressively) using $K$ learned query vectors:

$$\mathbf{Q} = [\mathbf{q}_0, \mathbf{q}_1, \dots, \mathbf{q}_{K-1}] \in \mathbb{R}^{K \times d_{\text{model}}}$$

These queries interact via multi-head self-attention and cross-attention against the observation memory $\mathbf{M}$ across $L_{\text{dec}}$ decoder layers (standard: $L_{\text{dec}} = 6$ or $8$, $n_{\text{heads}} = 8$, $d_{\text{model}} = 512$):

$$\mathbf{H}^{(l)} = \operatorname{CrossAttention}\left( \operatorname{SelfAttention}\left(\mathbf{H}^{(l-1)}\right),\, \mathbf{M} \right)$$

The final representations $\mathbf{H}^{(L_{\text{dec}})} \in \mathbb{R}^{K \times d_{\text{model}}}$ are mapped to robot action dimensions via a linear output head:

$$\hat{\mathbf{A}}_t = \mathbf{H}^{(L_{\text{dec}})} \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \in \mathbb{R}^{K \times d_a}$$

---

## 4. Inference Mechanics & Temporal Ensembling

At inference time, the CVAE encoder $q_\phi$ is **discarded**. The policy operates in evaluation mode by setting the latent variable deterministically to the prior mean:

$$z^* = \mathbb{E}_{p(z)}[z] = \mathbf{0}_{d_z}$$

Optionally, stochastic rollouts can sample $z \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ to explore alternative trajectories if the robot becomes stuck.

### 4.1 Chunk Boundary Discontinuity
If an action chunk $\mathbf{A}_t = [a_t, \dots, a_{t+K-1}]$ is executed open-loop for all $K$ steps before querying the policy again at $t+K$, two major failure modes occur:
1. **Open-Loop Divergence**: The robot cannot react to dynamic changes or external disturbances occurring within the $K$-step window.
2. **Chunk Boundary Jitter**: Independent chunks $\mathbf{A}_t$ and $\mathbf{A}_{t+K}$ often exhibit subtle velocity or acceleration discontinuities at the boundary $a_{t+K-1} \to a_{t+K}$, inducing physical jerk, actuator wear, and dropped objects.

### 4.2 Temporal Ensembling via Exponential Moving Average (EMA)
To ensure continuous closed-loop reactivity while preserving smooth transitions, ACT queries the policy at **every single timestep** $t = 0, 1, 2, \dots$ at high frequency ($50\,\text{Hz}$).

At step $t$, the policy generates a new chunk $\mathbf{A}_t \in \mathbb{R}^{K \times d_a}$. Consequently, there exist up to $K$ concurrent predictions for the action at timestep $t$, originating from prior model invocations at $t, t-1, t-2, \dots, t - \min(t, K-1)$:

```
Time Step:          t-2       t-1        t        t+1       t+2
Chunk from t-2:   [ a_0  ,   a_1   ,   a_2*  ,   a_3   ,   a_4   ]
Chunk from t-1:             [ a_0  ,   a_1*  ,   a_2   ,   a_3   ]
Chunk from t:                       [ a_0*  ,   a_1   ,   a_2   ]
                                         |
                                         v
Ensemble Action at time t:   a_t = w_0*a_0* + w_1*a_1* + w_2*a_2* / sum(w)
```

The actual action sent to the robot servos is computed as a weighted exponential average over all active predictions:

$$a_t = \frac{\sum_{i=0}^{N_t - 1} w_i \cdot \mathbf{A}_{t-i}[i]}{\sum_{i=0}^{N_t - 1} w_i}$$

where:
* $N_t = \min(t + 1, K)$ is the number of active overlapping predictions.
* $\mathbf{A}_{t-i}[i]$ denotes the $i$-th element of the chunk predicted at timestep $t-i$ (corresponding to target time $(t-i) + i = t$).
* $w_i$ is an exponentially decaying weighting coefficient:

$$w_i = \exp(-m \cdot i), \quad m \ge 0$$

### 4.3 Mathematical Analysis of the Decay Parameter $m$
The decay parameter $m$ regulates the trade-off between **immediacy** (responsiveness to fresh observations) and **temporal smoothness** (consistency with past commitments):

* **As $m \to \infty$ (Greedy Receding Horizon):**
  $$w_0 \gg w_{i \ge 1} \implies a_t \approx \mathbf{A}_t[0]$$
  The policy becomes purely closed-loop, executing only the first action of the latest chunk. While highly reactive, this eliminates the temporal regularization benefits of chunking and re-introduces high-frequency jitter.

* **As $m \to 0$ (Uniform Trajectory Averaging):**
  $$w_i = 1 \implies a_t = \frac{1}{N_t} \sum_{i=0}^{N_t - 1} \mathbf{A}_{t-i}[i]$$
  All historical predictions receive equal weight. This maximizes motion smoothness but introduces phase lag, reducing responsiveness to rapid visual changes.

* **Optimal Operational Regime ($m \in [0.01, 0.1]$):**  
  The effective memory horizon $\tau_{1/2}$ (half-life) of an action commitment satisfies:
  $$\tau_{1/2} = \frac{\ln(2)}{m} \approx \frac{0.693}{m}$$
  For $m = 0.05$, $\tau_{1/2} \approx 14$ steps ($280\,\text{ms}$ at $50\,\text{Hz}$), providing an ideal blend of trajectory stability and closed-loop adaptability.

---

## 5. Algorithmic Pseudocode

### 5.1 Training Loop (PyTorch / LeRobot Standard)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def act_loss_step(
    policy_encoder: nn.Module,
    policy_decoder: nn.Module,
    batch: dict,
    beta: float = 10.0
) -> torch.Tensor:
    """
    Computes ACT CVAE training loss over an action chunk.
    batch['observation.images']: Dict of camera tensors (B, C, H, W)
    batch['observation.state']: Current joint positions (B, d_a)
    batch['action']: Ground-truth action chunk (B, K, d_a)
    """
    images = batch["observation.images"]
    qpos = batch["observation.state"]
    action_chunk = batch["action"]  # shape (B, K, d_a)

    # 1. Forward pass through CVAE Encoder (training only)
    # Output: latent distribution parameters mu and log_var
    mu, log_var = policy_encoder(action_chunk, qpos)  # shapes (B, d_z)

    # 2. Reparameterization trick
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std

    # 3. Policy Decoder forward pass
    # Generates predicted trajectory chunk (B, K, d_a)
    pred_chunk = policy_decoder(images, qpos, z)

    # 4. Reconstruction loss (L1 norm over trajectory)
    l1_loss = F.l1_loss(pred_chunk, action_chunk, reduction="none")
    recon_loss = l1_loss.sum(dim=-1).mean()  # Sum over dims, average over batch & K

    # 5. KL Divergence: D_KL( N(mu, sigma^2) || N(0, I) )
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

    # 6. Total combined loss
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
```

### 5.2 Closed-Loop Temporal Ensembling Inference

```python
import numpy as np

class TemporalEnsembler:
    """
    Maintains a rolling buffer of predicted action chunks and computes
    exponentially weighted ensemble actions at 50Hz.
    """
    def __init__(self, chunk_size: int = 50, action_dim: int = 7, m: float = 0.05):
        self.K = chunk_size
        self.d_a = action_dim
        self.m = m
        # Buffer stores chunks: list of tuples (arrival_time, chunk_array)
        self.active_chunks = []
        self.step_idx = 0

    def update(self, new_chunk: np.ndarray) -> np.ndarray:
        """
        new_chunk: np.ndarray of shape (K, d_a) predicted at current step_idx.
        Returns: filtered action vector of shape (d_a,).
        """
        self.active_chunks.append((self.step_idx, new_chunk))
        
        # Remove expired chunks (older than K steps)
        self.active_chunks = [
            (t0, chunk) for (t0, chunk) in self.active_chunks
            if (self.step_idx - t0) < self.K
        ]

        numerator = np.zeros(self.d_a, dtype=np.float32)
        denominator = 0.0

        for t0, chunk in self.active_chunks:
            offset = self.step_idx - t0  # index within this chunk
            weight = np.exp(-self.m * offset)
            numerator += weight * chunk[offset]
            denominator += weight

        smoothed_action = numerator / denominator
        self.step_idx += 1
        return smoothed_action
```

---

## 6. Inductive Biases, Strengths & Failure Modes

| Dimension | Characteristic | Impact on Embodied Manipulation |
| :--- | :--- | :--- |
| **Primary Inductive Bias** | Explicit joint trajectory correlation across time ($K$ steps) | Eliminates high-frequency chatter; preserves kinematic continuity. |
| **Multimodality Handling** | Conditional VAE with Gaussian latent space $\mathbb{R}^{d_z}$ | Captures discrete modes, but susceptible to posterior collapse if $\beta$ is miscalibrated. |
| **Inference Latency** | Single forward pass through non-autoregressive transformer ($< 25\,\text{ms}$) | Operates natively at $50\,\text{Hz}$ on consumer GPUs (RTX 3070 / 4090). |
| **Sample Efficiency** | High ($\sim 50$ teleoperated demonstrations suffice for complex tasks) | Ideal for rapid real-world prototyping on low-cost hardware (ALOHA, SO-100). |
| **Failure Modes** | Visual covariate shift; compounding error if chunk horizon $K$ is too small | Background clutter or novel lighting causes latent drift; requires data augmentation. |
