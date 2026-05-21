---
type: paper
title: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
authors: ["Lucas Maes", "Quentin Le Lidec", "Damien Scieur", "Yann LeCun", "Randall Balestriero"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2603.19312
rw_id: 01ks5nwzpjcrjxth0p8ss96tfs
topics: [jepa, world-models, self-supervised-learning, latent-dynamics, model-predictive-control, representation-collapse]
priority: high
read_state: queued
relevance: ""
added: 2026-05-21
last_updated: 2026-05-21
---

# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

**TL;DR** — LeWM is the first JEPA that trains *stably* end-to-end from raw pixels with just two loss terms: a next-embedding prediction MSE and **SIGReg**, a regulariser that enforces isotropic Gaussian latent embeddings to prevent collapse. No stop-gradient, no EMA, no pretrained encoder, no reconstruction loss. At 15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than DINO-WM while remaining competitive on PushT / OGBench-Cube / Reacher / Two-Room, and latent probes show physical quantities (agent location, block position, block angle) are recoverable. The headline structural simplification is going from PLDM's seven-term VICReg-derived loss to a two-term loss with a single effective hyperparameter λ.

## Context & motivation

Joint Embedding Predictive Architectures (JEPAs) "offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse" (§1).

The paper organises the prior landscape into three categories (Fig. 2):

- **End-to-end methods (PLDM)** — learn encoder and predictor jointly from pixels, but "require many hyperparameters and lack formal collapse guarantees."
- **Foundation-based methods (DINO-WM)** — freeze a pretrained vision encoder (DINOv2) to avoid collapse, forgoing end-to-end learning.
- **Task-specific methods (Dreamer, TD-MPC)** — require reward signals or privileged state access during training.

LeWM positions itself as the only method that is simultaneously "end-to-end, task-agnostic, pixel-based, reconstruction- and reward-free, and requires only a single hyperparameter with provable anti-collapse guarantees" (§1).

Setting: fully offline, reward-free, trained on unannotated trajectories of observations and actions from "behavior policies with no optimality requirements; they may be pseudo-expert or exploratory, as long as they sufficiently cover the environment dynamics" (§3.1).

## Method

### Problem formulation
Given trajectories of length $T$ of raw pixel observations $o_{1:T}$ and actions $a_{1:T}$, learn an encoder and a predictor such that the latent dynamics predict the next-frame embedding without collapse.

### Architecture (§3.1, "Model Architecture")

$$\text{Encoder: } z_t = \text{enc}_\theta(o_t)$$
$$\text{Predictor: } \hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t)$$

- **Encoder**: ViT-tiny by default — ~5M params, patch size 14, 12 layers, 3 attention heads, hidden dim 192. Embedding is the [CLS]-token of the last layer followed by a **1-layer MLP with Batch Normalization**. The paper is explicit that BN here is load-bearing: "This step is necessary because the final ViT layer applies a Layer Normalization, which prevents our anti-collapse objective from being optimized effectively" (§3.1).
- **Predictor**: 6-layer transformer, 16 attention heads, 10% dropout, ~10M params. Actions are conditioned via **AdaLN** (Adaptive Layer Norm) "applied at each layer", with AdaLN parameters "initialized to zero to stabilize training and ensure that action conditioning impacts the predictor training progressively" (§3.1).
- **History conditioning**: predictor takes $N$ past frame embeddings with causal masking and auto-regresses next embeddings.
- A projector with the same shape as the encoder's projection follows the predictor.
- Total: ~15M parameters.

### Core idea
A two-term loss is enough. The prediction term gives the encoder a "be predictable" pressure; SIGReg, a distributional regulariser, is what prevents the trivial collapse this pressure would otherwise create.

### Training objective (§3.1)

$$\mathcal{L}_{\text{pred}} \triangleq \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \qquad \hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t) \quad (1)$$

$$\text{SIGReg}(\mathbf{Z}) \triangleq \frac{1}{M}\sum_{m=1}^{M} T(h^{(m)}), \qquad h^{(m)} = \mathbf{Z}u^{(m)} \quad (2)$$

$$\mathcal{L}_{\text{LeWM}} \triangleq \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(\mathbf{Z}) \quad (3)$$

where $u^{(m)} \in \mathbb{S}^{d-1}$ are $M$ random unit-norm directions, $T(\cdot)$ is the **Epps–Pulley** univariate normality-test statistic, and $\mathbf{Z} \in \mathbb{R}^{N \times B \times d}$ are latent embeddings collected over history length, batch, and embedding dim. By the **Cramér–Wold theorem**, "matching all one-dimensional marginals is equivalent to matching the full joint distribution" — that's the formal claim behind SIGReg's collapse guarantee.

Hyperparameters: $M = 1024$ random projections, $\lambda = 0.1$ by default. The paper claims $M$ has "negligible impact on downstream performance," leaving λ as "the only effective hyperparameter to tune" — and notes λ can be searched by simple bisection (O(log n)) versus PLDM's seven-term landscape (described as $O(n^6)$).

### What is *not* used
No stop-gradient. No EMA target network. No pretrained encoder. No reconstruction loss. No reward signal.

### Inference / planning (§3.2)

At test time the model is used inside a **Model Predictive Control** (MPC) loop with the Cross-Entropy Method (CEM) as the inner solver. Given initial obs $o_1$ and goal obs $o_g$:

$$\hat{z}_{t+1} = \text{pred}_\phi(\hat{z}_t, a_t), \quad \hat{z}_1 = \text{enc}_\theta(o_1)$$

$$\mathcal{C}(\hat{z}_H) = \|\hat{z}_H - z_g\|_2^2, \quad z_g = \text{enc}_\theta(o_g) \quad (4)$$

$$a_{1:H}^* = \arg\min_{a_{1:H}} \mathcal{C}(\hat{z}_H) \quad (5)$$

Only the first $K$ actions of each planned sequence are executed before re-planning from the new observation. Planning horizon $H$ trades long lookahead against autoregressive error accumulation.

## Experimental setup (§4.1)

- **Environments**: Push-T (2D manipulation), OGBench-Cube (3D manipulation), Two-Room (2D navigation), Reacher (2-joint reaching). All continuous-action.
- **Baselines**:
  - **DINO-WM** — JEPA using frozen DINOv2 as encoder. Run without proprioceptive input "for a fair comparison, unless specified otherwise."
  - **PLDM** — VICReg-based end-to-end pixel JEPA with a seven-term loss.
  - **GCBC** — goal-conditioned behavioral cloning policy.
  - **GCIVL, GCIQL** — goal-conditioned offline RL baselines.
- "For each method, we keep the hyperparameters fixed across all environments" (§4.1).
- Exact dataset sizes, image resolutions, and pseudo-expert collection details point to App. D/E (not pulled in this triage).

## Key results

### Planning performance (Fig. 6, §4.2)
- **PushT**: LeWM (pixels-only) beats PLDM by "18% higher success rate" and *also* beats DINO-WM with proprioceptive input.
- **Reacher**: LeWM "consistently outperforms PLDM and DINO-WM."
- **OGBench-Cube**: "DINO-WM slightly outperforms LeWM, possibly due to the higher visual complexity and the 3D nature of the environment, which makes encoder training more challenging."
- **Two-Room**: LeWM *underperforms* PLDM and DINO-WM. The paper's stated hypothesis: "the low diversity and low intrinsic dimensionality of this dataset make it difficult for the encoder to match the isotropic Gaussian prior enforced by SIGReg in a high-dimensional latent space" (§4.2). This is flagged as "a potential limitation of the SIGReg regularization in very low-complexity environments."

### Planning speed (Fig. 3, §4.2)
- LeWM achieves "up to 48× faster" planning than DINO-WM, with full planning runs "in under one second."
- ~200× fewer tokens at the encoder output than DINO-WM is named as the source of the speedup.

### Physical structure of the latent space (§5.1, Table 1)
Linear and MLP probes for physical quantities on Push-T:
- **Agent Location**: linear MSE 0.052 (LeWM) vs 1.888 (DINO-WM) vs 0.090 (PLDM); linear $r$ 0.974 vs 0.977 vs 0.955.
- **Block Location**: linear MSE 0.007 (LeWM) vs 0.006 (DINO-WM) vs 0.122 (PLDM); linear $r$ 0.986 vs 0.997 vs 0.938.
- **Block Angle**: linear MSE 0.187 (LeWM) vs 0.050 (DINO-WM) vs 0.625 (PLDM); linear $r$ 0.902 vs 0.979 vs 0.745.
- Pattern: LeWM is "consistently competitive" with DINO-WM (which the paper attributes partly to DINOv2's ~124M-image pretraining advantage) and beats PLDM across the board.

### Decoded rollouts (§5.1, Fig. 8)
- A *post-hoc* decoder trained on top of the frozen latent reconstructs visual scenes — "even though no reconstruction loss is used during training."
- Limitation noted: "Some finer details … are not fully captured by LeWM; for instance, the angle of the end-effector in OGBench-Cube" (Fig. 7).

### Temporal latent path straightening (§5.1, App. H)
- "LeWM's latent trajectories become increasingly straight on PushT over training as a purely emergent phenomenon, without any explicit regularization encouraging this behavior."
- Crucially: "LeWM achieves higher temporal straightness than PLDM, despite PLDM employing a dedicated temporal smoothness regularization term."

### Violation-of-expectation surprise detection (§5.2, Fig. 10)
- Tested across TwoRoom, PushT, OGBench-Cube with visual perturbations (colour swap) and physical perturbations (object teleportation).
- "LeWM consistently assigns higher surprise to frames containing physical violations compared to their unperturbed counterparts."

## Ablations (§4.3, App. G)

- **SIGReg internals (M projections, integration knots)**: "The performance is largely unaffected by these quantities, indicating that they do not require careful tuning." Practical implication: λ is the only hyperparameter that matters.
- **Hyperparameter search complexity**: bisection on the single λ ⇒ O(log n); PLDM's six free loss-weights ⇒ "polynomial time (O(n^6))."
- **Embedding dimensionality**: "performance quickly saturates beyond a certain threshold" — robust to encoder capacity choice.
- **Encoder architecture**: ViT-tiny default vs ResNet-18 (Tab. 8) — "competitive performance with both architectures, indicating that it is largely agnostic to the choice of vision encoder."
- **Training-curve smoothness** (Fig. 18 vs 19): LeWM's two-term loss is "smooth and monotonic"; PLDM's seven-term loss is "noisy and non-monotonic across several of its loss components."

## Limitations

Stated by the authors (§6, "Limitations & Future Work"):
- "Planning with current latent world models remains restricted to short horizons. Hierarchical world modeling represents a promising direction to address long-horizon" planning.
- SIGReg can hurt in very low-intrinsic-dimensionality environments (Two-Room result, §4.2).
- Decoded rollouts miss fine details like end-effector angle in 3D scenes (§5.1).

[analyst's view] Other limitations a careful reader would flag:
- All evaluation is on goal-reaching control tasks. No stochastic-environment evaluation, no high-stochasticity domains — failure modes of next-embedding-prediction under stochastic dynamics aren't tested.
- The "no EMA, no stop-gradient" claim is structurally elegant but heavily reliant on SIGReg actually delivering its collapse guarantee in practice. At scale, batch-statistics-based normality tests can become problematic.
- The 15M-parameter, single-GPU positioning is a competitive feature but also caps what this paper can say about scaling.

## Why it matters [analyst's view]

Three things stand out:

1. **The two-term loss is the contribution.** PLDM's seven-term VICReg loss has been a sore point for JEPA practitioners — hyperparameter search is genuinely painful. Going to *one effective hyperparameter* with a *formal* Cramér–Wold-backed collapse guarantee is the kind of simplification that gets adopted. The bisection-search argument (O(log n) vs O(n^6)) is doing real work.
2. **SIGReg is a structural alternative to the EMA / stop-gradient orthodoxy.** Most successful JEPA recipes (I-JEPA, V-JEPA) rely on EMA target encoders + stop-gradients to stabilise — tricks that "do not in general correspond to the minimization of a well-defined objective" (§2). LeWM's pitch is: replace the trick with a distributional matching objective, and the gradient flows everywhere. This is a cleaner story for anyone trying to prove things about JEPA training dynamics.
3. **The temporal-straightening emergence claim is the most intriguing result.** Latent paths straighten without explicit pressure to do so, and end up *straighter* than PLDM's explicitly-regularised paths. That's the same geometric instinct as [[papers/huang-2026-semantic-tube-prediction]]'s Geodesic Hypothesis — and it's emerging from the same author cluster (LeCun + Balestriero). Two simultaneous papers converging on "smooth low-curvature latent trajectories" as the right form for self-supervised representations is a signal worth tracking.

The result is also a direct empirical reply to [[papers/joseph-2026-physics-video-world-models]]: yes, world models *can* encode recoverable physical quantities (LeWM's probing table) — but whether the encoding is factorised vs. distributed is exactly what Joseph et al. measure for video models. Repeating that interpretability protocol on LeWM is the natural next step.

## Open questions

[analyst's view]
- **SIGReg + low intrinsic dimensionality.** The Two-Room failure is honest and instructive. Is there a learned-prior alternative (mixture of Gaussians, learned base measure) that doesn't punish low-D environments?
- **Action-conditioning via AdaLN-zero-init.** This is a quietly load-bearing design choice borrowed from diffusion-transformer literature. How much of the stability is AdaLN-zero-init vs SIGReg vs the BN-projection?
- **What does SIGReg do on text?** STP ([[papers/huang-2026-semantic-tube-prediction]]) regularises sequence-trajectory shape; LeWM regularises latent marginal distribution. Are these substitutable, complementary, or solving different collapse modes?
- **Scaling.** The 15M / single-GPU framing is a strength for accessibility but begs the question — does the two-term loss stay clean at 1B+ params, or do new pathologies appear?
- **Compositionality / multi-task.** All experiments are single-task. Does the latent admit task-mixing without re-training the encoder?
- **Stochastic environments.** Next-embedding MSE in a stochastic world is mean-prediction, which often collapses to averages. LeWM's experiments are deterministic — what breaks first under noise?

## Connections

- **Builds on / closest prior**: PLDM (the seven-term VICReg end-to-end JEPA that LeWM directly displaces) and DINO-WM (the frozen-DINOv2 alternative).
- **Concurrent / cousin work**:
  - [[papers/huang-2026-semantic-tube-prediction]] — same author cluster (LeCun + Balestriero); converges on geometric-shape priors for representation trajectories.
  - [[papers/joseph-2026-physics-video-world-models]] — interpretability protocol that is the natural next test for LeWM's latent.
  - [[papers/guo-2022-byol-explore]] — earlier self-predictive RL using BYOL-style stabilisation (EMA + stop-gradient); LeWM is partly a structural answer to "do we need EMA at all?"
  - [[papers/khetarpal-2024-byol-ac]] — action-conditional self-predictive theory; relevant to LeWM's predictor design.
  - [[papers/higuera-2026-visuo-tactile-world-models]] — world model for contact-rich tasks via tactile fusion; a complementary modality-augmentation answer to the "what should a world model condition on?" question.
- **Topic MOCs**: [[topics/jepa]], [[topics/world-models]], [[topics/self-predictive-learning]]
- **Author indices**: [[authors/lucas-maes]]

## Selected quotes

> "We introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings." — abstract

> "This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative." — abstract

> "LeWM's latent trajectories become increasingly straight on PushT over training as a purely emergent phenomenon, without any explicit regularization encouraging this behavior … LeWM achieves higher temporal straightness than PLDM, despite PLDM employing a dedicated temporal smoothness regularization term." — §5.1

> "By the Cramér–Wold theorem, matching all one-dimensional marginals is equivalent to matching the full joint distribution." — §3.1
