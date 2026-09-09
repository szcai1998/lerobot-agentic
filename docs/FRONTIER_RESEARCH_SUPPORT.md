# Frontier Research Support: Embodied AI Reliability, Recovery, and VLA Systems

**Document role:** Living research-intelligence support for the engineering project  
**Status:** **FROZEN LANDSCAPE SNAPSHOT v1.0 — audited 2026-09-08**; future landscape changes create a new dated version rather than mutating this snapshot  
**Audit baseline:** 2026-09-08  
**Primary purpose:** Keep the engineering project anchored to current embodied-AI research without turning the project itself into a full robotics PhD thesis.

---

## 1. Purpose

This file answers:

> **What does the current research frontier already contain, what resources should the project reuse, what claims are unsafe, and where is there still useful research headroom?**

It does **not** define the engineering system. The stable engineering synthesis is `LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md`; implementation sequencing belongs separately in `ROADMAP.md`.

The project should use this file as:

- a map of current papers, benchmarks, official frameworks, and public repositories;
- a source of strong baseline candidates;
- a claim-control mechanism;
- a warning system for ideas that have already been published;
- a weekly research radar for changes that materially affect the engineering project;
- support for a later technical report / arXiv-style preprint.

The engineering project should **reuse the frontier**, not reproduce the entire literature.

### 1.1 Freeze Semantics

This file is frozen as a **dated research-landscape snapshot**, not as a permanent claim about what will remain current. The snapshot exists so that later engineering decisions and public claims can be traced to the exact state of the field used when the architecture was frozen.

- This v1.0 snapshot is valid **as of 2026-09-08**.
- New papers, repositories, model releases, or benchmark changes do not rewrite this snapshot; they produce a new dated revision.
- Before any final public research claim, the latest required audit must still be run.
- A frozen snapshot can remain historically correct even after it stops being the latest landscape.

---

## 2. Operating Principle: Guilty Until Proven

No idea in this project may be called:

- novel;
- SOTA;
- first;
- frontier;
- general;
- policy-agnostic;
- benchmark-leading;

until the claim is supported by a dated audit of primary sources.

A sophisticated implementation is not automatically a scientific contribution.

### Evidence classes

| Class | Meaning | Allowed use |
|---|---|---|
| **E0** | Official framework/model/benchmark docs or official repository | Engineering contract, supported feature, released checkpoint |
| **E1** | Peer-reviewed paper | Strong scientific baseline / established prior art |
| **E2** | Recent preprint | Frontier tracking; provisional scientific evidence |
| **E3** | Public reproducible repository / live research prototype | Engineering precedent and novelty threat |
| **E4** | Secondary article, unsupported inference, assumed hardware number | Not admissible in frozen engineering/research claims |

### 2.1 Source Verification Contract

A resource enters the **active engineering plan** only after verification of the properties relevant to its use:

```yaml
resource: ...
url_resolves: true
identity_matches_claimed_work: true
paper_status: peer_reviewed | preprint | repository | official_docs
code_available: true | false | partial
weights_available: true | false | n/a
data_available: true | false | n/a
license_checked: true | false
benchmark_compatible: true | false | unknown
reproduction_status: verified | planned | blocked | reference_only
last_verified: YYYY-MM-DD
```

Rules:

- a paper citation does not prove code exists;
- a project page does not prove assets are downloadable;
- a repository does not prove reported experiments reproduce;
- an arXiv paper is not labeled peer-reviewed unless a venue is independently verified;
- hardware numbers are accepted only from official documentation or our measurements;
- a source that becomes unavailable is demoted until reverified.

---

## 3. Project-Level Research Position

The engineering project is **not** intended to invent another foundation VLA.

Its research support target is:

> **Runtime reliability for modern robot policies: failure detection, recovery, test-time computation, outcome verification, and learning from execution experience.**

The current narrow research case study, if it survives future audits, is:

> **Can recovery knowledge learned around one frozen VLA transfer usefully to a second heterogeneous VLA when the recovery layer uses policy-independent observable execution evidence and physically verified outcomes rather than architecture-specific hidden states?**

This is a **candidate investigation**, not a guaranteed novelty claim.

The repository remains valuable even if this research hypothesis fails.

---

## 4. What Is Already Baseline-Level

The following are not acceptable standalone novelty claims in 2026:

| Idea | Status | Representative prior art |
|---|---|---|
| High-level VLM/VLA planner controlling a low-level motor policy | established | Gemini Robotics ER 2, RT-H, GHOST |
| Event-triggered replanning | occupied | VLA-Corrector, FLARE, TOWN-VLA |
| Action-chunk reset/truncation | occupied | VLA-Corrector |
| Adaptive execution horizon | occupied | VLA-Corrector |
| Failure-aware retry | occupied | FAR |
| Retry/reset hierarchy | occupied | FLARE |
| Learned VLA failure detection | occupied | SAFE, FailureSpot, Foresight |
| Hidden-state probing for failure/correction | occupied | ProbeAct |
| Sample-and-verify test-time action selection | occupied | RoboMonkey, FM-Steer |
| Selective expensive reasoning / slow path | occupied | TOWN-VLA |
| World-model-guided candidate search | occupied family | tau0-VLA, WAM test-time scaling, EA-WM |
| Stale-observation world-model correction | occupied | DreamActVLA |
| Residual recovery around frozen VLA | occupied | ReCoVLA |
| Recovery orchestration around off-the-shelf policy | occupied | RoboBRIDGE |
| Guarded executable-skill/VLA agent loop with prerequisite checks, outcome verification, recovery, structured trajectories, and swappable low-level policies | occupied | EmbodiedSkills (arXiv 2609.01281) |
| Cost-aware selection among recovery mechanisms | direct live collision | Recovery-Selection public project |
| Learning from robot failures/corrections | occupied family | IntervenGen, FailSafe, Dream2Fix, FAR, RECAP, RedFlow |
| Memory that tracks verified task progress | occupied direction | AGM, RoboMME, PI memory work |
| High standard-LIBERO score as proof of frontier intelligence | invalid inference | benchmark saturation / validity work |
| Deployment-facing runtime-fidelity / recovery-aware evaluation protocol | occupied | ROEP (Sensors 2026) |
| Structured VLA failure taxonomy + recovery-aware metrics on low-cost robot benchmark | occupied | SO-101 failure/recovery benchmark (arXiv 2606.08881) |

---

## 5. Mandatory Resource Map

The project does **not** need to reproduce everything below.  
The implementer **does** need to understand where the engineering choices come from.

### 5.1 Core framework and official engineering sources

| Resource | Why it matters |
|---|---|
| Hugging Face LeRobot stable documentation | Dataset, policy, processor, training, evaluation contracts |
| AllenAI `vla-evaluation-harness` (`vla-eval`) | Unified model/benchmark evaluation substrate with Docker isolation, model servers, sync/live runners, recording, and broad benchmark coverage; preferred upstream evaluation substrate where compatible |
| LeRobot `main` documentation | Frontier integrations not yet present in stable |
| LeRobot hardware guide | Single-GPU feasibility and training envelope |
| LeRobot ACT | Historical/local action-chunking baseline |
| LeRobot SmolVLA | Lightweight modifiable VLA |
| LeRobot pi0.5 | Strong modern VLA reference |
| LeRobot MolmoAct2 | Strong high-capacity policy with documented memory measurements |
| LeRobot RoboCasa365 integration | Current non-LIBERO evaluation path |

Official links:

- https://huggingface.co/docs/lerobot/
- https://github.com/allenai/vla-evaluation-harness
- https://arxiv.org/abs/2603.13966
- https://huggingface.co/docs/lerobot/main/hardware_guide
- https://huggingface.co/docs/lerobot/act
- https://huggingface.co/docs/lerobot/smolvla
- https://huggingface.co/docs/lerobot/pi05
- https://huggingface.co/docs/lerobot/molmoact2
- https://huggingface.co/docs/lerobot/main/robocasa

### 5.2 Benchmarks

| Benchmark | Role in this project | Notes |
|---|---|---|
| **Standard LIBERO** | compatibility only | Near saturation; not primary scientific evidence |
| **LIBERO-plus** | static robustness | Useful for controlled visual/state perturbations |
| **LIBERO-RECOVER** | emerging recovery benchmark | Use only after official assets/protocol/license are verified |
| **RoboCasa365** | primary non-LIBERO confirmation | 365 tasks; broad kitchen diversity; still substantial headroom |
| **Custom MuJoCo microbenchmark** | mechanism/debug sandbox | Engineering foundation, not publication benchmark |
| RoboMME | optional memory validation | Only if memory becomes a primary claim |
| VLABench | optional long-horizon validation | Stretch, not mandatory |
| LIBERO-Safety | orthogonal safety track | Only if project introduces a safety claim |

Required links:

- https://huggingface.co/docs/lerobot/libero
- https://huggingface.co/docs/lerobot/libero_plus
- https://arxiv.org/abs/2609.05178
- https://liulin815.github.io/LIBERO-Recovery/
- https://huggingface.co/docs/lerobot/main/robocasa
- https://robocasa.ai/leaderboard.html
- https://arxiv.org/abs/2606.04233

### 5.3 Failure detection and runtime correction

Minimum literature awareness:

- SAFE — learned VLA failure detection  
  https://github.com/vla-safe/SAFE
- FailureSpot — timestamp-level failure localization  
  https://arxiv.org/abs/2609.04277
- Foresight — action-conditioned world-model failure detection with cross-policy analysis  
  https://arxiv.org/abs/2606.23085
- ProbeAct — hidden-state probes and intervention  
  https://arxiv.org/abs/2606.09740
- VLA-Corrector — detection, chunk truncation, adaptive horizons, corrective inference  
  https://arxiv.org/abs/2607.01804
- FAR — failure-aware retry and experience-based improvement  
  https://arxiv.org/abs/2607.01111
- FLARE — retry/reset hierarchy with multimodal supervision  
  https://arxiv.org/abs/2608.26645
- ReCoVLA — residual recovery around frozen VLA  
  https://arxiv.org/abs/2606.09630
- TOWN-VLA — selective slow-path intervention  
  https://arxiv.org/abs/2608.23224
- EmbodiedSkills — unified VLA-agent framework with prerequisite checks, bounded low-level VLA execution, post-action outcome verification, structured trajectories, recovery, and swappable/adaptable low-level policies  
  https://arxiv.org/abs/2609.01281
- FPC-VLA — peer-reviewed supervisor-based failure prediction/correction  
  https://www.sciencedirect.com/science/article/pii/S095741742600655X
- ValueFormer — policy-independent online value/mistake signal  
  https://arxiv.org/abs/2608.02958
- OmniTacTune — policy-independent residual adaptation across heterogeneous policies  
  https://arxiv.org/abs/2607.03723

### 5.4 Test-time computation / verification

- RoboMonkey — sample-and-verify test-time scaling  
  https://proceedings.mlr.press/v305/kwok25a.html
- FM-Steer — value-guided action selection  
  https://openaccess.thecvf.com/content/CVPR2026/html/Song_FM-Steer_Enhance_Generalist_Policies_with_Value-Guided_Cascaded_Denoising_CVPR_2026_paper.html
- E-TTS — history-aware reasoning/action test-time scaling  
  https://arxiv.org/abs/2606.27268
- tau0-VLA — world-model-guided hierarchical test-time computation  
  https://arxiv.org/abs/2608.16885
- DreamActVLA — stale-observation / asynchronous correction  
  https://dream-act-vla.github.io/

### 5.5 Failure data and learning from experience

- IntervenGen  
  https://arxiv.org/abs/2405.01472
- MimicGen  
  https://github.com/NVlabs/mimicgen
- FailSafe  
  https://arxiv.org/abs/2510.01642
- Dream2Fix  
  https://arxiv.org/abs/2603.13528
- RePO-VLA  
  https://arxiv.org/abs/2605.09410
- RedFlow  
  https://arxiv.org/abs/2607.27782
- PI RECAP / pi*0.6  
  https://www.pi.website/blog/pistar06

### 5.6 Memory / verified progress

- RoboMME  
  https://robomme.github.io/
- Achievement-Grounded Memory (AGM)  
  https://arxiv.org/abs/2608.29537
- Physical Intelligence memory research  
  https://www.pi.website/research/memory

Critical design lesson:

```text
attempted action != achieved state
attempted recovery != successful recovery
```

Any project memory should distinguish commands from verified physical outcomes.

### 5.7 Adjacent Engineering and Evaluation Infrastructure

These projects constrain the engineering claim even when they are not direct algorithmic baselines:

- **LeRobot FAR** — LeRobot-based remote VLA deployment measurement and failure-aware recovery for SmolVLA/π0.5, with LIBERO and SO-101 evaluation, asynchronous client/server timing, RTC interaction, and proprioceptive grasp-failure recovery.  
  https://github.com/HyuanTan/lerobot_far
- **AllenAI `vla-evaluation-harness` (`vla-eval`)** — a unified executable VLA evaluation framework that decouples models from benchmarks, runs benchmarks in Docker, exposes model servers, provides sync/live runners, episode recording, reproducibility tooling, and broad benchmark coverage. Generic model/benchmark decoupling, Dockerized evaluation, and cross-benchmark execution are therefore **upstream infrastructure to reuse**, not LeRobot Reliability Lab contributions.  
  https://github.com/allenai/vla-evaluation-harness
- **RoboBRIDGE** — modular monitor/perceptor/planner/controller/robot-interface orchestration around off-the-shelf VLAs with hierarchical recovery across LIBERO, RoboCasa, and real-world cases.  
  https://arxiv.org/abs/2607.27881
- **EmbodiedSkills** — guarded VLA-agent runtime with executable-skill contracts, prerequisite checking, bounded VLA execution, post-action verification, recovery, structured trajectories, and replaceable/adaptable low-level policies. Outcome verification and a fixed swappable agent interface are therefore prior art, not our novelty.  
  https://arxiv.org/abs/2609.01281
- **ROEP** — peer-reviewed deployment-facing VLA evaluation protocol covering runtime-interface fidelity, checkpoint provenance, perturbation repeatability, failure semantics, and recovery/shield-support evidence. These evaluation concepts are prior art, not our novelty.  
  https://doi.org/10.3390/s26154757
- **Benchmarking VLA Models on SO-101: Failure and Recovery Analysis** — low-cost real-robot benchmark with a structured failure taxonomy, semantic/execution failure decomposition, and recovery-aware metrics across π0.5, SmolVLA, Wall-X, and ACT.  
  https://arxiv.org/abs/2606.08881

**Engineering consequence:** LeRobot Reliability Lab must differentiate on the reliability lifecycle and evidence contract—checkpoint/processor integrity, runtime-vs-oracle separation, standardized failure/recovery telemetry, explicit achievement verification, reproducible run bundles, cost accounting, and comparable recovery interfaces—not on generic orchestration or generic multi-benchmark evaluation.

---

## 6. Live Novelty-Threat Register

These resources matter even if they are not peer-reviewed.

| Threat | Why it matters | Current response |
|---|---|---|
| Recovery-Selection | Cost-aware selection among heterogeneous recoveries | Broad selector novelty withdrawn |
| HARP-VLA | Runtime instability + routing among recovery depths | Keep in weekly audit |
| TOWN-VLA | Selective expensive intervention around frozen VLA | Semantic slow path = baseline |
| Foresight | Cross-policy failure detection | Detection transfer != recovery transfer |
| ReCoVLA | Recovery around multiple VLA architectures | Cross-policy protocol must be materially distinct |
| RoboBRIDGE | Modular monitor/perceptor/planner/controller orchestration around off-the-shelf VLAs with hierarchical recovery | Do not claim generic orchestration as the engineering or scientific contribution |
| EmbodiedSkills | Executable-skill contract with preconditions, bounded VLA execution, post-action verification, structured trajectories, recovery, and swappable low-level policies | Do not claim guarded execution, outcome verification, or a fixed swappable VLA-agent interface as new |
| LeRobot FAR | LeRobot-specific async pipeline measurement + failure-aware recovery for SmolVLA/π0.5 on LIBERO/SO-101 | Do not claim first LeRobot recovery/runtime-reliability tooling; differentiate on shared reliability/evidence contracts |
| AllenAI `vla-evaluation-harness` | Unified executable VLA evaluation framework with model/benchmark decoupling, Docker isolation, live runners, recording, and broad benchmark coverage | Reuse as preferred evaluation substrate where compatible; do not rebuild generic benchmark/model abstraction |
| ROEP | Runtime-fidelity/checkpoint-provenance/perturbation/failure-semantics and recovery-support evaluation protocol | Do not claim that deployment-facing reliability evidence or checkpoint provenance checking is conceptually new |
| SO-101 failure/recovery benchmark | Structured failure taxonomy and recovery-aware metrics across several VLA/IL policies | Do not claim first structured VLA failure taxonomy or first recovery-aware VLA evaluation metrics |
| OmniTacTune | Policy-independent residual adaptation | Cannot claim generic policy-independent correction |
| ValueFormer | Policy-independent online value signal | Outcome/value model = infrastructure |
| FPC-VLA | Peer-reviewed supervisor/corrector | Semantic supervisor = baseline |

Template for repository file `docs/frontier_claim_ledger.md`:

```yaml
claim_id: C-001
claim: "..."
status: candidate | baseline_only | collision | rejected | verified
as_of: 2026-09-08
last_checked: 2026-09-08
next_check_due: 2026-09-15
primary_sources:
  - ...
closest_prior_art:
  - ...
closest_live_threat:
  - ...
what_is_still_distinct: ...
engineering_impact_if_invalidated: low | medium | high
```

---

## 7. Benchmark Guidance

### Standard LIBERO

Use only to prove:

- policy checkpoint is loaded correctly;
- observation/action preprocessing matches upstream;
- our evaluation harness is not broken.

Do not optimize the project around moving from 97% to 98%.

### LIBERO-plus

Use for:

- static visual/state robustness;
- regression tests;
- controlled robustness curves.

### LIBERO-RECOVER

Preferred failure-recovery evidence if:

1. official scenarios are available;
2. evaluation scripts are reproducible;
3. license is usable;
4. recovery scenario semantics can be integrated without modifying the intended benchmark.

If those conditions fail, label any implementation:

```text
LIBERO-Recover-inspired
```

not official LIBERO-RECOVER.

### RoboCasa365

Use as non-LIBERO confirmation because:

- current LeRobot `main` integrates it;
- it exposes 365 tasks across broad environments;
- atomic and composite tasks are available;
- current benchmark headroom remains substantial;
- it reduces dependence on one benchmark family.

Do **not** run all 365 tasks.

Freeze a representative task subset before final results.

---

## 8. 4090-First Model Guidance

### Hardware rule

The engineering project is designed for:

> **1 × NVIDIA RTX 4090, 24 GB VRAM**

Official LeRobot hardware guidance describes 24 GB consumer GPUs as comfortable for:

- light behavior cloning;
- diffusion policies;
- SmolVLA;

and tight for large VLA training at batch size 1.

Therefore large-model full fine-tuning is not a mandatory deliverable.

### Required policy roles

| Role | Preferred policy | Requirement |
|---|---|---|
| Historical/local sanity | ACT | mandatory foundation only |
| Modifiable modern VLA | **SmolVLA** | mandatory |
| Strong frozen reference | **MolmoAct2 or pi0.5**, chosen by measured feasibility | mandatory one-of |
| Predictive/world-model | one verifier/reference path | optional |

### MolmoAct2

Official LeRobot measurements report approximately:

- inference: 12.1 GiB;
- action-expert fine-tuning, batch 8: 16.5 GiB;
- action-expert fine-tuning, batch 32: 21.4 GiB;
- LoRA-VLM, batch 8: 20.2 GiB;
- full fine-tuning: >48 GiB.

These measurements were collected on H100 and must **not** be assumed identical on a 4090.

Engineering implication:

- inference is a strong local candidate;
- action-expert adaptation is an optional measured experiment;
- full model training is out of scope.

### pi0.5

Official LeRobot LIBERO full fine-tuning examples are sized for an 80 GB GPU.

Engineering implication:

- local inference: benchmark empirically;
- expert-only adaptation: optional if it fits;
- full fine-tuning: not required;
- do not make the entire project depend on pi0.5 local training.

---

## 9. Research Questions Worth Preserving

The research-support file should keep a small number of questions alive rather than turning each into an implementation mandate.

### RQ-A — Cross-policy recovery transfer

Can a recovery layer learned around Policy A improve Policy B without requiring Policy-B-specific hidden representations?

### RQ-B — Achievement-grounded memory

Does storing **verified outcomes** outperform chronological memory of attempted actions?

### RQ-C — Recovery cost

When does a cheap local correction dominate an expensive semantic/world-model recovery?

### RQ-D — Failure heterogeneity

Which recovery mechanisms are effective for:

- perceptual failure;
- grasp/contact failure;
- stale observation;
- semantic goal failure;
- action execution corruption?

### RQ-E — Outcome adaptation

Can realized recovery outcomes improve the recovery layer under a bounded interaction budget without retraining the base policy?

Only **one** of these needs to become the core technical-report hypothesis.

---

## 10. What the Engineering Project May Safely Claim

Without any new algorithmic result, the public repository may claim:

- integration of multiple LeRobot policy families behind a common policy adapter;
- integration of multiple manipulation benchmark families;
- reproducible failure/recovery evaluation;
- checkpoint integrity verification;
- standardized recovery telemetry;
- modular failure detector / recovery expert / outcome verifier interfaces;
- 4090-first reproducibility;
- comparative engineering study of current recovery mechanisms.

It may **not** claim:

- SOTA recovery;
- first VLA recovery system;
- first adaptive horizon;
- first failure detector;
- first test-time recovery;
- first cost-aware router;
- policy-agnostic recovery without actual cross-policy evidence;
- generalization without non-LIBERO evidence.

---

## 11. Research-Support Existence Test

The research-support file must continuously test whether the engineering project still deserves to exist.

### The repository still has a reason to exist if

- current recovery methods remain fragmented across incompatible implementations;
- the project adds a reliability lifecycle **on top of** upstream evaluation infrastructure rather than rebuilding `vla-eval`'s generic model/benchmark matrix;
- LeRobot does not already expose one common reliability/evaluation contract across the selected policies and benchmarks;
- checkpoint/revision/normalization integrity remains a real reproducibility problem;
- standardized recovery telemetry and outcome verification reduce meaningful evaluation work;
- single-4090 reproducibility is useful to practitioners who cannot depend on accelerator clusters;
- the repo remains useful even after every candidate novelty claim is removed.

### The repository should pivot or shrink if

- an upstream LeRobot release provides the same reliability middleware and evaluation workflow at equal or better quality;
- RoboBRIDGE, EmbodiedSkills, LeRobot FAR, AllenAI VLA Evaluation Harness, LeRobot upstream, or another maintained project already provides the same complete reliability lifecycle with equivalent or better policy/benchmark coverage, integrity checks, intervention telemetry, outcome verification, run-bundle reproducibility, and usability;
- our implementation becomes mostly adapters around existing APIs with no meaningful reduction in engineering/evaluation effort;
- maintaining multiple benchmark integrations costs more than the insight they provide;
- the research question starts dictating unnecessary infrastructure.

The correct response to overlap is **reuse, integrate, or narrow** — not manufacture novelty.

---

## 12. Weekly Frontier Audit Protocol

### Every 7 days

Search the previous 14 days for:

```text
VLA failure recovery
vision language action recovery
robot failure detector
cross-policy robot recovery
test-time scaling VLA
world action model recovery
robot recovery benchmark
LIBERO recovery
RoboCasa recovery
embodied memory recovery
```

Also inspect:

- LeRobot `main` release/docs changes;
- relevant new official GitHub repositories;
- benchmark leaderboard/release changes.

### At every engineering gate that changes scientific claims

Run a full previous-60-day search.

### Before final technical-report experiments

Run a 30-day exact-claim audit and freeze the claim ledger.

### Stop condition

If a new work directly demonstrates the intended research hypothesis with equal or stronger evidence:

1. demote the idea;
2. preserve the engineering work;
3. redefine the technical-report question;
4. do **not** redesign stable repository infrastructure unless necessary.

This is the purpose of separating this file from the engineering specification.

---

## 13. Frontier Support Definition of Done

This document is doing its job when:

- current strong baselines are known;
- old ideas are clearly demoted;
- new papers can be inserted without rewriting the project architecture;
- every scientific claim has an evidence state;
- the engineering team knows which external components should be reused;
- the public report can explain why each baseline and benchmark was selected.

It is **not** a static literature review.

It is the project's research navigation layer.


---

## 14. Freeze Audit Record

The v1.0 snapshot was re-audited on **2026-09-08** after the engineering technical report was finalized. The following high-risk landscape claims were externally rechecked against current primary or direct project sources:

| Claim / resource | Re-audit result | Freeze treatment |
|---|---|---|
| RoboBRIDGE occupies generic VLA orchestration + hierarchical recovery | confirmed by arXiv/project description | treat generic orchestration as occupied |
| EmbodiedSkills occupies guarded executable-skill orchestration, prerequisite checks, outcome verification, structured trajectories, recovery, and swappable low-level VLA interface | confirmed by arXiv 2609.01281 (2026-09-01) | treat these runtime/verification/interface ideas as prior art |
| LeRobot FAR provides LeRobot-specific pipeline measurement and failure recovery for SmolVLA/π0.5 | confirmed by public repository and experiment documentation | treat first-LeRobot-recovery/tooling claim as occupied |
| AllenAI `vla-evaluation-harness` provides executable model/benchmark decoupling, Dockerized evaluation, live runners, recording, and broad benchmark coverage | confirmed by official repository/docs | reuse it as preferred evaluation substrate; treat generic cross-benchmark execution as occupied |
| LeRobot 24 GB guidance: SmolVLA practical, large VLAs tight | confirmed by current official hardware guide | retain 4090-first constraint with measured feasibility |
| FPC-VLA is peer reviewed | confirmed in *Expert Systems with Applications* (2026) | retain as established supervisor/corrector prior art |
| RoboMonkey is peer reviewed | confirmed in CoRL 2025 / PMLR | retain as established test-time sampling/verifier prior art |
| ROEP provides deployment-facing runtime-fidelity/checkpoint-provenance/recovery-support evaluation | confirmed in peer-reviewed *Sensors* 2026 | treat these evaluation ideas as prior art; reuse/compare protocol concepts |
| SO-101 benchmark includes structured failure taxonomy and recovery-aware metrics | confirmed in arXiv 2606.08881 | failure taxonomy and recovery metrics are not novelty claims |

No re-audited item invalidated the project architecture. The audit **narrowed engineering claims** but did not eliminate the reliability/evidence-contract justification.

---

## 15. Current Approval Status

**As of 2026-09-08:**

- The engineering direction is **approved as a candidate public reliability/evaluation project**.
- The project is **not approved to claim SOTA or first-of-kind recovery**.
- Generic VLA orchestration is already occupied by adjacent systems such as RoboBRIDGE.
- Guarded executable-skill loops, prerequisite checking, outcome verification, structured trajectories, and swappable low-level VLA interfaces are already occupied by EmbodiedSkills.
- Generic cost-aware recovery selection is already threatened directly by Recovery-Selection.
- First-of-kind LeRobot recovery tooling is not claimable because LeRobot FAR already exists.
- Generic multi-benchmark VLA evaluation is not claimable because broad evaluation harnesses already exist.
- Runtime-fidelity/checkpoint-provenance/recovery-aware evaluation concepts are not claimable as new because ROEP already formalizes them.
- Structured VLA failure taxonomy and recovery-aware metrics are not claimable as first-of-kind because the SO-101 failure/recovery benchmark already exists.
- The cross-policy recovery-transfer case study remains **optional candidate research**, not the justification for building the repository.
- The repository must satisfy the existence and engineering-completion tests in `LEROBOT_RELIABILITY_LAB_TECHNICAL_REPORT.md` before optional research complexity is allowed to define the project.

The project's strongest durable justification is:

> **reproducible, 4090-accessible, LeRobot-centered reliability evaluation and recovery middleware with policy/benchmark manifests, standardized telemetry, outcome verification, and comparative recovery experiments.**
