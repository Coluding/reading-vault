---
type: paper
title: "Dual-Rate Diffusion: Accelerating diffusion models with an interleaved heavy-light network"
authors: ["Grigory Bartosh", "David Ruhe", "Emiel Hoogeboom", "Jonathan Heek", "Thomas Mensink", "Tim Salimans"]
year: 2026
venue: "arXiv:2605.18190"
url: https://arxiv.org/abs/2605.18190
rw_id: 01kszyjtfsjdxsn0s9x5fyzt25
topics: [generative-models, diffusion-efficiency]
priority: medium
read_state: skimmed
relevance: ""
added: 2026-06-03
last_updated: 2026-06-03
---

# Dual-Rate Diffusion: Accelerating diffusion models with an interleaved heavy-light network

## TL;DR

Dual-Rate Diffusion cuts diffusion inference cost by splitting the denoiser into
**two networks running at two rates**: a heavy **context encoder** evaluated only
every $K$-th step to extract high-dimensional global features, and a **light
denoising model** evaluated at *every* step that reuses those features to refine
local detail. The intuition is spectral: global (low-frequency) structure is
established early and stays stable, so re-computing it with a heavy net at every
step wastes capacity. On ImageNet 64×64 / 128×128 it matches or beats standard
diffusion FID at **2–4× lower FLOPs**, and composes with **Moment Matching
Distillation** for few-step generation (student even beats teacher). Work done at
Google DeepMind Amsterdam (Bartosh interning from UvA).

## Context & motivation

Diffusion's inference cost is dominated by repeated evaluation of one heavy
network. Two prior lines attack this: (1) fewer steps (distillation, few-step
training, ODE solvers) — but the *per-step* network stays heavy; (2) reuse of
earlier-step computation (self-conditioning, DeepCache, Clockwork Diffusion,
token caching) — suggesting a high-capacity net isn't needed at every step.
Dual-Rate is **orthogonal to step-reduction**: it lowers cost *per step* by
structurally decoupling global vs local computation into two separate models,
rather than managing layer/token-wise caching schedules inside one network.

Two framings motivate it: (a) conditioning the light denoiser on encoder outputs
generalizes **self-conditioning** (Chen et al. 2022) beyond data-space; (b) in
the **frequency domain** (Dieleman 2024; Rissanen 2023), global low frequencies
emerge first and remain stable while local detail emerges late — so delegate
global structure to one sparse heavy pass and let the light net handle the rest.

## Method

### Problem formulation
Standard variance-preserving diffusion: $\mathbf{z}_t = \alpha_t \mathbf{x} + \sigma_t \varepsilon$,
reverse process predicts $\hat{\mathbf{x}}$ and samples a less-noisy state. The
training objective is the usual diffusion ELBO with sigmoid log-SNR weighting.

### Core idea
Two interleaved networks at different evaluation rates. Context encoder
$E_\Theta(\mathbf{z}_\tau, \tau) = \mathbf{e}$ run every $K$-th step; light
denoiser $g_\theta(\mathbf{z}_t, t, \mathbf{e})$ run every step using the most
recent feature $\mathbf{e}$ from the nearest preceding heavy step $\tau$.

### Architecture / algorithm
Both nets are **UVit** (from Hoogeboom et al. 2025 "Simpler Diffusion"); the
denoiser is the same architecture with fewer blocks/channels — **~2× lighter in
FLOPs** than the encoder (64×64: 44.0 vs 108.5 GFLOPs; 128×128: 73.0 vs 137.4).
Features are taken from the last layer of *each* spatial-resolution level of the
encoder UVit and concatenated into the denoiser at the matching level (with a
linear projection to match dims). Two timesteps ($t$ and $\tau$) are injected via
**FiLM** conditioning. The training objective extends the standard loss to a
joint expectation over $(t, \tau)$:

$$D_{KL}(q(\mathbf{x})\,\|\,p_{\Theta,\theta}(\mathbf{x})) \leq \tfrac{1}{2}\,\mathbb{E}\!\left[-\tfrac{d\lambda_t}{dt}e^{\lambda_t}w(\lambda_t)\big\|\mathbf{x} - g_\theta(\mathbf{z}_t, t, E_\Theta(\mathbf{z}_\tau, \tau))\big\|^2\right]$$

A subtle theoretical point: the encoder + context state $\mathbf{z}_\tau$ are
**theoretically redundant** (the target process is Markovian, so a flexible
enough denoiser alone suffices), but a *light* denoiser can't approximate it well
— the encoder's job is to package information from $\tau$ for cheap reuse. Because
the denoiser depends on two states, **non-Markovian posteriors like DDIM cause a
train/inference mismatch**, so they use only Markovian processes throughout.

### Three training tricks (all help; Table 1)
1. **3-level conditioning** — feed encoder features from every hierarchy level, not just the last (1.30 → 1.24).
2. **Extra data augmentation** — random translation (1.30 → 1.26).
3. **Feature/embedding dropout** — randomly zero $\mathbf{e}$ with $p=0.5$, forcing additive, robust features. **Most impactful** (1.30 → 1.14). All three together → 1.12.

### Training procedure
v-parameterization, cosine schedule, Adam (lr $10^{-4}$, no weight decay),
gradient-clip norm 1, batch 1024, ~$10^6$ steps, TPU v6e. Classifier-free
guidance (drop class $p=0.1$); for Dual-Rate both nets evaluated twice
(cond/uncond).

### Distillation (Dual-Rate MMD)
Substitute the dual-rate architecture as the MMD student. Key change vs vanilla
MMD: intermediate states $\mathbf{z}_s$ must be generated by **rolling out the
student's light network** (not the teacher forward process) to keep training
in-distribution and avoid compounding errors. Student can't be initialized from
the teacher (architectures differ), so they init the encoder from the teacher and
**freeze it during a cheap pretraining phase**. Discrete (not continuous) time
sampling stabilizes training.

## Experimental setup

- **Datasets:** class-conditional ImageNet 64×64 and 128×128.
- **Metric:** FID on 50k samples vs training set; cost in TFLOPs and NFE.
- **Baselines:** VDM++, EDM2-XL, SiD2 / their own base model (FID 1.25 @ 64×64); for distillation: CTM, sCD-XXL, DMD2, standard Moment Matching Distillation.

## Key results

- **Standard diffusion, 64×64:** base model FID **1.25 at 55.5 TFLOPs**; Dual-Rate ($K{=}16$, 512 steps) FID **1.12 at 13.14 TFLOPs** — *better FID at ~4× less compute*. At 256 steps still FID 1.12 (7.44 TFLOPs); at 128 steps FID 1.52 (3.53 TFLOPs).
- **Standard diffusion, 128×128:** base FID 1.50 (70.37 TFLOPs) → Dual-Rate 1.48 at 28.23 TFLOPs (~2.5× less).
- **Dual-Rate MMD, 64×64:** at 8 steps ($K{=}8$) FID **1.17 at 0.79 TFLOPs**, beating standard MMD 8-step (1.24 @ 0.87). The **student beats the teacher** (2.38 → 1.17) even without teacher guidance.
- A naively-shrunk single baseline net gives a strictly worse compute/FID trade-off than Dual-Rate (Fig. 4).

## Ablations

- **Feature dropout** dominates the three training tricks (Table 1).
- **Heavy-step count / feature dim** (Figs 3, 6): reducing heavy steps or feature dim below ~8 sharply hurts; reducing *feature dimensionality* trades off far better than reducing *depth* (blocks).
- **Convolutional light net**: can be 20× cheaper than encoder but never reaches FID < 3 on 64×64 — UVit transformer blocks needed.
- **MMD full rollout** (Algo 4): generating $\mathbf{z}_\tau$ from the student too (vs standard noising) improves FID 1.61 → 1.17 at $K{=}4$, 8 steps; full rollout needs no data samples.

## Limitations

Paper's own: (1) **higher training cost per iteration** — both nets evaluated each
step (could be mitigated by alternating training, left to future work); (2)
dependence on two observations forces **Markovian-only** processes (no DDIM) and,
for MMD, a student rollout during training. [analyst] Gains are shown only on
pixel-space ImageNet with UVit; transfer to latent-space DiTs or text-to-image is
untested, and the optimal heavy-step *schedule* is hand-set (every $K$-th step).

## Why it matters [analyst's view]

A clean "two-speed" factorization of the denoiser that is **orthogonal** to the
dominant few-step/distillation race — and composes with it. The spectral framing
(global structure is slow, local detail is fast) gives it a principled story
rather than an ad-hoc caching heuristic, and the heavy/light decoupling is more
structurally honest than in-network caching (Clockwork, DeepCache). It sits in the
vault's [[topics/generative-models]] efficiency thread alongside
[[blogs/shing-diffusionblocks]] (block-wise diffusion) and
[[papers/yang-2026-replaid-continuous-diffusion]]. Interesting contrast with
[[papers/ding-2024-diffusion-world-model]]: both fight diffusion's per-call cost,
but DWM predicts the whole trajectory in one pass while Dual-Rate keeps many steps
but makes most of them cheap.

## Open questions / things to verify

- Does it transfer to **latent-space Diffusion Transformers** and text-to-image, or is it UVit/pixel-space specific?
- Can the heavy-step schedule be **learned/adaptive** rather than fixed every-$K$?
- The "video/audio with temporal redundancy" extension they flag — how much more speedup when redundancy is higher?
- Training-cost overhead vs the inference saving: what's the break-even for a deployed model?

## Connections

- Contrasts with: [[papers/ding-2024-diffusion-world-model]] (cheap-whole-trajectory vs many-cheap-steps)
- Topic MOCs: [[topics/generative-models]]
- Related: [[blogs/shing-diffusionblocks]], [[papers/yang-2026-replaid-continuous-diffusion]]
- Author indices: [[authors/grigory-bartosh]], [[authors/emiel-hoogeboom]], [[authors/tim-salimans]]

## Selected quotes

> "Evaluating the denoising model independently at every time step leads to repeated, redundant processing of this global structure, which consumes the model capacity. Dual-Rate Diffusion may be seen as delegating this task to the heavy context encoder once." — §1

> "Theoretically, the context encoder and the secondary observation $\mathbf{z}_\tau$ are redundant. … However, a light denoising model alone may struggle to approximate the target process well enough." — §3.1
