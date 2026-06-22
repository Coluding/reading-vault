---
type: paper
title: "Optimal Transport Flow Matching by Design"
authors: ["Shimon Malnick", "Matan Rusanovsky", "Ohad Fried", "Shai Avidan"]
year: 2026
venue: arXiv (preprint)
url: https://arxiv.org/abs/2606.04092
rw_id: 01ktspdnv04p2dryetc15r45q4
topics: [flow-matching, optimal-transport, generative-models, diffusion-models, distillation]
priority: high
read_state: queued
relevance: ""
added: 2026-06-10
last_updated: 2026-06-22
---

# Optimal Transport Flow Matching by Design

> The paper's running title on the PDF is **"Designing Optimal Transport Flows"** (project page: malnick.net/designing_ot_flows). Same work.

## TL;DR

Flow matching trains a velocity field on independently-paired noise/data
samples, which forces the field to average over crossing transport paths and
yields curved trajectories that need many ODE steps. The optimal-transport
(OT) coupling would straighten them, but computing OT against a Gaussian prior
is intractable ($O(n^3)$ time, $O(n^2)$ memory). The key move here is to stop
*solving* OT and instead *design the prior so its identity coupling to the data
is already OT-optimal*. They show that **low-frequency projections of natural
images** are such a prior: because natural-image spectra decay as $1/f^2$, a
low-pass/downsampled version of each image is close to it in $\ell_2$, so
pairing each image with its own low-frequency version is (empirically) the OT
coupling — no solver, no per-batch or dataset-wide precomputation. They
interpolate this prior with Gaussian noise ($\alpha=0.5$) to avoid manifold
collapse/overfitting while *preserving* the OT-identity coupling (verified by
Hungarian assignment: 99.9% preserved on CIFAR-10, 100% on FFHQ latents up to
$\alpha\approx0.5$). Result: **>2× lower trajectory curvature** than the best
baseline and better FID in the few-step regime (e.g. 20% better than minibatch
OT-FM and 38% better than independent FM at 1 step on FFHQ), with **no changes
to the flow model**. It composes with latent diffusion, classifier-free
guidance, and one-step frameworks like MeanFlow.

## Context & motivation

The ideal generator would be a single map from noise to data — one evaluation,
no iterative solving. That map is the optimal transport map, and if we had it,
generation would be one step. We don't: computing OT between a high-dimensional
data distribution and a Gaussian prior is computationally prohibitive (exact
solvers are cubic in sample count; entropic relaxations [Cuturi 2013] remain
infeasible at modern dimensionalities).

Flow matching [Lipman et al. 2023; Liu et al. 2023; Albergo & Vanden-Eijnden]
sidesteps this: instead of computing the transport map, learn a velocity field
that continuously deforms the prior into the data over $t\in[0,1]$. Standard
practice samples noise and data **independently** at training time, forgoing any
global coupling. This works remarkably well but inherits a tension — because the
true OT coupling is unknown, independent pairing makes the learned field average
over conflicting transport directions, producing **curved, crossing
trajectories** that demand many integration steps. Straighter trajectories =
fewer steps, so trajectory straightening is the central lever for few-/single-
step generation.

Two prior lines attack this. (1) **OT-based pairing**: minibatch-OT / multisample
FM [Pooladian et al.; Tong et al.] solve small transport problems per batch;
semi-discrete methods (AlignFlow) and dataset-wide assignments solve larger ones
— all at the cost of per-batch or per-dataset OT solves, and bias or overhead.
(2) **Alternative priors**: Cold Diffusion (deterministic degradations: blur,
downsample, mask), frequency-aware destruction, blur↔noise interpolation,
Fourier-magnitude corruption, anisotropic Gaussians, time-varying blue noise.
This paper sits in the second camp but with a sharper claim: choose the prior so
that the **identity coupling itself is OT-optimal**, getting straight
trajectories *without solving or approximating any OT problem*. Their Table 1
positions the method as the only one that is simultaneously from-scratch,
OT-based, scales with batch $B$, scales with data $N$, and is dataset-agnostic.

## Method

### Problem formulation
Flow matching over base $p_0$ (noise) and data $p_1$ on $\mathbb{R}^d$. Learn a
time-dependent flow $\phi_t:\mathbb{R}^d\to\mathbb{R}^d$, $t\in[0,1]$, with
$\phi_0(x_0)=x_0$ and $\phi_1$ pushing $p_0$ forward to $p_1$. Writing
$x_t=\phi_t(x_0)$, the flow obeys the ODE

$$\frac{d}{dt}x_t = u_t(x_t),\tag{1}$$

where $u_t$ is the velocity field generating the flow. The goal is a neural
field $v_\theta(x_t,t)\approx u_t$.

### Background: the standard FM objective and why paths cross
Direct regression onto $u_t$ is intractable, but the **conditional** objective is
equivalent [Lipman et al.]. Under the linear-interpolation path

$$\phi_t(x_0\mid x_1) = x_t = t\,x_1 + (1-t)\,x_0,\tag{2}$$

with $x_0\sim p_0$, $x_1\sim p_1$ sampled **independently**, the conditional
velocity is $u_t(x_t\mid x_1) = x_1 - x_0$, giving the loss

$$\mathcal{L}(\theta) = \mathbb{E}_{t,x_0,x_1}\Big[\,\lVert v_\theta(x_t,t) - (x_1 - x_0)\rVert_2^2\,\Big],\quad t\sim U[0,1].\tag{3}$$

Each conditional velocity $x_1-x_0$ is a straight path for one *specific* pair,
but because $x_0,x_1$ are drawn independently, different pairs' straight paths
**cross**. The learned $v_\theta$ must average conflicting directions → curved
inference trajectories → many ODE steps.

### Why OT helps (and why it's intractable)
For a map $T$, let $T_\#(p_1)$ denote the push-forward. The Monge OT problem with
squared cost is

$$\inf_{T:\,T_\#(p_1)=p_0}\;\mathbb{E}_{x_1\sim p_1}\big[\lVert T(x_1)-x_1\rVert^2\big].\tag{4}$$

Replacing independent coupling in (3) with the OT coupling removes crossings: any
crossing could be "uncrossed" to lower total squared cost, contradicting
optimality. So OT pairing ⇒ straighter trajectories ⇒ fewer steps. But the OT map
between data and a Gaussian, while guaranteed to exist [Brenier], costs $O(n^3)$
time / $O(n^2)$ memory — infeasible at scale.

### Core idea
**Treat the prior as a design choice, not a fixed input.** Once the prior is free
to choose, the OT coupling between prior and data is no longer unique: *many*
priors admit an OT-optimal **identity** coupling to the data. Pick one that is
also (a) cheap to sample and (b) structured. Their choice: the **low-frequency
projection of each image**.

### Designing the OT prior
Seek a transform $T:\mathbb{R}^d\to\mathbb{R}^d$ defining the prior
$\tilde p_0 = T_\#(p_1)$, with two goals:

1. **Identity coupling is OT-optimal.** Under the squared cost of (4), the
   identity coupling pays exactly the mean squared reconstruction error of $T$.
   So we want $T$ with *small reconstruction error* on $p_1$. Strict orthogonal
   projections make the identity coupling *exactly* OT-optimal (Prop. D.2 in the
   paper).
2. **Expressive yet lightweight to sample.** The prior must carry enough
   structure to support generating $p_1$, but be simple enough that a small model
   can sample it.

**Why low frequencies.** Natural-image power spectra decay as $1/f^2$ [Field;
Ruderman], so low-pass filtering retains most of each image's $\ell_2$ energy —
small reconstruction error (goal 1) — while the result is a coarse low-res image,
simple for a lightweight model to generate (goal 2). Crucially, the two goals are
*independent*: Fourier truncation, random pixel masking, and random patch masking
are all strict orthogonal projections (so all preserve the OT-identity coupling),
**but only the low-frequency variants give low FID and low curvature** under noise
perturbation (Tab. C.3). OT-optimality alone is *not sufficient* — the prior must
also retain low-frequency structure a small model can sample. Among low-frequency
options, **bicubic downsampling** beats Fourier truncation slightly and is
adopted. Conceptually this makes the coarse-to-fine principle of cascaded /
latent / pyramid diffusion *explicit at the level of the flow's coupling*: the
flow never invents coarse structure — its start point already has it — and the
residual job is high-frequency synthesis, exactly where the OT-identity coupling
holds.

### Noise interpolation (the load-bearing fix)
The deterministic coupling $x_1\mapsto T(x_1)$ is the exact coupling sought, but
it concentrates the prior on a low-dimensional subspace: nearby prior points can
map to distant data points, so $v_\theta$ again averages conflicting directions
→ curvature. It also shows the model the same pair every epoch (overfitting).
Fix: interpolate the low-frequency representation with Gaussian noise

$$x_0 = (1-\alpha)\,T(x_1) + \alpha\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I),\tag{5}$$

with $\alpha\in[0,1]$. $\alpha=0$ = deterministic coupling; $\alpha=1$ = standard
FM with random pairing. The expected squared distance between two noisy prior
samples decomposes as

$$\mathbb{E}\big[\lVert x_0^{(i)}-x_0^{(j)}\rVert^2\big] = (1-\alpha)^2\,\mathbb{E}\big[\lVert T(x_1^{(i)})-T(x_1^{(j)})\rVert^2\big] + \alpha^2\,\mathbb{E}\big[\lVert\epsilon_i-\epsilon_j\rVert^2\big].\tag{6}$$

The first term is mean pairwise distance in the projected subspace; the second
evaluates to $2\alpha^2 d$, **growing linearly with dimensionality** — so even
moderate noise separates prior samples substantially. The question is whether the
OT-identity coupling *survives* this noise. They compute the exact discrete OT
assignment (Hungarian algorithm) between 10,000 noisy prior samples and their data
samples across noise levels and measure the fraction matching the identity
coupling: it stays OT-optimal up to $\alpha\approx0.5$ — **99.9% preserved in
pixel space (CIFAR-10), 100% in latent space (FFHQ)** — and only beyond
$\alpha=0.5$ does noise disrupt the structure. FID is minimized near
$\alpha=0.5$ (improves $\alpha{=}0\to0.5$, degrades after), so they fix
$\alpha=0.5$ everywhere. The $\alpha=0$ endpoint is itself a clean baseline (pure
coarse-to-fine), so the $0\to0.5$ FID gap isolates the contribution of
OT-preserving noise interpolation beyond the coarse-to-fine substrate.

### Pipeline and training
Instantiate $T = U\circ D$: $D:\mathbb{R}^d\to\mathbb{R}^{d'}$ downsamples
($x_1^\downarrow = D(x_1)$), $U$ upsamples back to $\mathbb{R}^d$.
- **Train $v_\theta$** (Fig. 2a): for each data sample compute $T(x_1)$, perturb
  via (5) to get $x_0$, pair with $x_1$, run standard FM training of $v_\theta$
  (Alg. A.1). **No architectural change.**
- **Train $G_\phi$** separately (Fig. 2b): a *lightweight* flow-matching generator
  mapping Gaussian noise to the low-frequency space $\mathbb{R}^{d'}$, i.e. it
  learns to sample $\tilde p_0$ (coarse generation) at a small fraction of the
  full dimensionality.
- **Inference** (Fig. 2c): $G_\phi$ produces $\hat x_1^\downarrow$, deterministically
  upsampled to $U(\hat x_1^\downarrow)$ and noise-perturbed to form $x_0$; then
  solve the ODE (1) with $v_\theta$ to synthesize high-frequency detail.

### Cost accounting (effective NFE)
The two-stage pipeline adds the $G_\phi$ cost. One $G_\phi$ step ≈ $0.09\times$ a
main-model step. They use 8 $G_\phi$ steps by default → 1 main-model step =
**effective NFE 1.72**; with a single $G_\phi$ step the overhead is only 0.09
(effective NFE 1.09). All results report effective NFE.

## Experimental setup

- **Datasets:** CIFAR-10 (32×32, pixel space), FFHQ (256×256), ImageNet (256×256).
  256² datasets trained with SiT in the latent space of a pretrained autoencoder
  (32×32×4 latents), downsampled 4× per spatial dim → 8×8 low-frequency prior.
  CIFAR-10 follows the Tong et al. recipe.
- **Baselines:** IFM (independent coupling), OT-FM (minibatch OT, Tong et al.),
  AlignFlow (semi-discrete OT). All single-network, **same architecture and
  parameter count** as $v_\theta$. On ImageNet they add classifier-free guidance
  and integrate the prior into **MeanFlow** (one-step framework).
- **Metrics:** FID (log scale in plots) for quality; **curvature** [Liu et al.;
  Lee et al.] for trajectory straightness.

## Key results

*Summarized — see the paper's figures for full FID-vs-NFE curves.*

- **CIFAR-10 (pixel):** beats IFM and OT-FM at all step counts; beats AlignFlow at
  4 and 8 effective NFE (and AlignFlow needs a full-dataset semi-discrete OT
  pre-solve). Curvature **≥2× below AlignFlow** (second best) and **4× below IFM**.
  Methods converge as steps grow — the advantage is concentrated in the few-step
  regime.
- **FFHQ 256² (latent):** best FID at all step counts except NFE=2 (where effective
  cost is only 1.72). Largest gains at 1 step: **+20% over OT-FM, +38% over IFM.**
  Curvature **0.083** vs OT-FM 0.172 vs IFM 0.201 (≥2× better).
- **ImageNet 256² (partial training, CFG):** FID 314.80→**248.42** with CFG and
  322.18→**268.14** without, at 1 NFE (1.09 effective); curvature reduced >2.3×.
  Absolute FIDs are high because training was truncated for compute; the
  comparison is controlled (same arch/budget).
- **MeanFlow integration (ImageNet):** MeanFlow+Ours at 1.72 effective NFE beats
  MeanFlow at both 1 and 2 steps (FID 54.66 vs 59.07/55.40), curvature 0.097 vs
  0.221 (2.3× straighter) — gains hold *even inside a framework already optimized
  for one-step generation*.

## Ablations

- **Projection operator (Sec. C.1):** the central ablation. Strict orthogonal
  projections (Fourier truncation, random pixel/patch masking) all preserve OT-
  optimality of the identity coupling, but only **low-frequency** variants give low
  FID + low curvature → OT-optimality alone is insufficient; low-frequency
  structure (samplable by a light model) is the necessary extra ingredient.
- **Downsampling factor (Sec. C.2)** and **$G_\phi$ step count (Sec. C.3)**: shows
  robustness; 8 $G_\phi$ steps adopted.
- **Noise ratio $\alpha$ (Sec. 4.1 / Fig. B.3):** FID minimized at $\alpha=0.5$;
  OT coupling preserved up to ~0.5.

## Limitations

- **The pipeline is not fully OT-coupled (paper's own).** The OT-identity coupling
  holds for the flow $\tilde p_0\to$ data, but **not** for generating $\tilde p_0$
  itself — $G_\phi$ is trained under *independent* coupling. So the whole pipeline
  isn't end-to-end OT.
- **$G_\phi$ is the bottleneck.** An oracle CIFAR-10 experiment (replace $G_\phi$
  with true low-frequency projections of training images) improves 1-NFE FID from
  **90.16 → 28.12** — a huge gap, meaning final quality is gated by coarse-prior
  generation. Upside: $G_\phi$ is decoupled, so any improvement to it directly
  improves the pipeline.
- **Domain specificity (paper's own):** relies on the $1/f^2$ spectrum of *natural
  images*; not obviously transferable to data without low-frequency-concentrated
  structure.
- **[analyst's view]** Absolute ImageNet FIDs are uninterpretable in isolation
  (partial training); the claims rest on *relative* curvature/FID at matched
  budget, which is fair but means "SOTA generation" is not demonstrated — "SOTA
  trajectory straightening at matched cost" is.

## Why it matters [analyst's view]

This is a genuinely elegant reframing: the field spent years building cheaper OT
*solvers* (minibatch, semi-discrete, Sinkhorn), and this paper asks whether you
need to solve OT at all if you can *choose a prior whose trivial coupling is
already optimal*. The insight that "design the prior" decouples OT-optimality from
samplability — and that low-frequency projection happens to satisfy both — is the
kind of move that's obvious only in hindsight. It also reframes
cascaded/coarse-to-fine generation (which the community treats as an architecture
trick) as an OT-coupling statement, which is a cleaner mental model.

In the vault this sits squarely with [[topics/optimal-transport]] and
[[topics/flow-matching]], and connects directly to two adjacent notes:
[[blogs/jiha-autoregression-vs-diffusion]] frames the AR-vs-diffusion gap through
*the same* OT-coupling/path-crossing lens, and [[blogs/dieleman-diffusion-integral]]
surveys the few-step-sampling literature (consistency, MeanFlow, shortcut models)
that this method explicitly *composes with* rather than competes against. The key
contrast: those few-step methods modify the **training objective** (average vs
instantaneous velocity, consistency constraints); this modifies the **coupling**
the velocity field is trained on — orthogonal axes, hence composable (demonstrated
via MeanFlow). That orthogonality is the most reusable idea here.

## Open questions / things to verify

- Does the low-frequency-prior trick survive to data without $1/f^2$ spectra
  (audio? video? non-natural images? latents of non-image modalities)? The FFHQ
  *latent*-space result is encouraging — the OT-identity coupling held on
  autoencoder latents — but that's still image-derived.
- How much of the headline gain is the OT coupling vs simply *starting closer to
  the data* (the $\alpha=0$ coarse-to-fine baseline already does well)? The paper
  isolates this with the $0\to0.5$ FID gap but it'd be worth seeing the decomposition
  on ImageNet too.
- Can $G_\phi$ be folded into a single OT-coupled stage to close the
  not-fully-OT-coupled limitation (e.g. recursive application of the same prior
  design at coarser scales — a true OT cascade)?
- Exact-OT verification used Hungarian on 10k samples; does identity-coupling
  preservation hold at much larger sample counts / higher resolution?

## Connections

- Builds on: standard flow matching (Lipman et al. 2023 *Flow Matching*; Liu et al.
  2023 *Rectified Flow*; Albergo & Vanden-Eijnden *Stochastic Interpolants*) and the
  conditional-FM objective.
- Contrasts with: minibatch/multisample OT-FM (Tong et al.; Pooladian et al.),
  semi-discrete AlignFlow — all *solve/approximate* OT; this *designs it away*.
- Complements (composable): MeanFlow, consistency models, shortcut models,
  Reflow/distillation — objective-side few-step methods. See
  [[blogs/dieleman-diffusion-integral]].
- Related priors: Cold Diffusion, frequency-aware destruction, blur↔noise
  interpolation, cascaded / latent / pyramid diffusion.
- Topic MOCs: [[topics/flow-matching]], [[topics/optimal-transport]],
  [[topics/generative-models]], [[topics/diffusion-models]], [[topics/distillation]]
- Related notes: [[blogs/jiha-autoregression-vs-diffusion]] (same OT/path-crossing
  lens), [[blogs/dieleman-diffusion-integral]] (few-step sampling survey),
  [[papers/litman-2026-attention-priors]] (also recasts a mechanism via entropic OT)
- Author indices: [[authors/shimon-malnick]], [[authors/matan-rusanovsky]],
  [[authors/ohad-fried]], [[authors/shai-avidan]]

## Selected quotes

> "Rather than solving for the OT coupling, we reformulate the problem. Once the prior is treated as a design choice rather than a fixed input, the OT coupling between prior and data is no longer unique." — Abstract

> "OT-optimality of the identity coupling is therefore not sufficient, the prior must also retain low-frequency structure that a lightweight model can sample efficiently." — §4.1

> "The flow model never has to invent coarse structure, its starting point already contains it, and the residual task of synthesizing high-frequency detail is exactly the regime where the OT-identity coupling holds." — §4.1
