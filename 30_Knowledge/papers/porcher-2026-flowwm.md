---
type: paper
title: Flow Matching in Feature Space for Stochastic World Modeling
authors:
  - François Porcher
  - Nicolas Carion
  - Karteek Alahari
  - Shizhe Chen
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2606.29059
rw_id: 01kwz6a5a0aybz926grrr614d2
topics:
  - flow-matching
  - world-models
  - generative-models
priority: high
read_state: done
relevance: ""
added: 2026-07-08
last_updated: 2026-07-15
---

## TL;DR

**FlowWM** is a stochastic visual world model that runs **flow matching directly inside the high-dimensional feature space of a frozen pretrained encoder (DINOv3)**, rather than in a low-dimensional VAE latent. The motivation is a genuine dilemma the paper proves formally (Appendix A): deterministic feature-space predictors trained with $\ell_1/\ell_2$ losses collapse a multimodal future onto its conditional mean/median — a point that corresponds to *no valid future* — while VAE-based stochastic models fix the multimodality but throw away the semantic structure needed for downstream perception. FlowWM keeps both stochasticity and semantic richness by generating in DINOv3 space, and it contributes the design recipe needed to make flow matching stable at $D=384$: a **shallow-but-wide projection head** (width 1024 > latent dim), **timestep-schedule shifting** toward noisier steps, and a **differentiable one-step projection** that supplies temporal-consistency and task-driven (detector) supervision without backpropagating through the ODE. On a synthetic Bouncing-Shapes benchmark it cuts F1 error from the prior stochastic baseline's **14.4 to 4.31**; on a real-world Waymo-derived **FuturePerception** benchmark it lifts object-detection $\text{AP}_L(3)$ from ~16.5 to **20.9** and improves depth and FVD (152.4 → 87.3, a 43% FVD gain over the deterministic predictor).

## Context & motivation

World models forecast future states from past context (Ha & Schmidhuber 2018). For driving and robotics, the paper argues the choice of *representation* is central: predicted states must preserve information for downstream perception/planning (detection, tracking, depth), not just look plausible. Two prevailing families each fail one goal. (1) **Video generative models** (diffusion / flow matching à la Sora, Wan, Open-Sora) first compress video into a low-dim spatiotemporal **VAE** latent, then model dynamics there — but VAE latents are optimized for pixel reconstruction, discarding geometry/semantics and hurting dense perception. (2) **Feature-space world models** (DINO-WM, DINO-Foresight) predict in pretrained encoder features (DINOv3, V-JEPA-2), which are semantically rich — but they are **deterministic**, yielding a single future, which is a critical flaw when the future is genuinely multimodal (a car may brake or continue; a pedestrian may cross or stop; occluded objects may reappear). The closest prior work, **Walker et al. (2025)**, does apply latent diffusion to frozen features, but its focus is comparing frozen representations, not improving the stochastic model; and naively porting the standard DiT recipe (tuned for low-dim VAE latents) to a much higher-dimensional space is unstable/suboptimal (Mousakhan et al. 2025; Zheng et al. 2025). FlowWM's stated contribution is to fix stochastic world modeling *in high-dimensional feature space*.

## Method

### Problem formulation
Given a video $I_{1:T}$, encode the first $T_{\text{context}}$ frames frame-by-frame with a **frozen** encoder $E$ (DINOv3): $x_{\text{ctx}} = x_{1:T_{\text{context}}} = E(I_{1:T_{\text{context}}}) \in \mathbb{R}^{T_{\text{context}} \times H \times W \times D}$ (spatial $H, W$, channel $D$). The model predicts future latents $\hat{x}_{\text{future}} = \hat{x}_{T_{\text{context}}+1:T} \in \mathbb{R}^{T_{\text{target}} \times H \times W \times D}$. Evaluation applies **frozen pretrained decoders** (a detector, a depth head) to the predicted features — explicitly probing preservation of object-centric information and scene geometry rather than pixel-level reconstruction.

### Core idea
Do flow matching **in the pretrained feature space itself** (semantic + stochastic), and make it work at high $D$ via architectural and scheduling choices plus a cheap differentiable endpoint estimate that lets you add temporal-consistency and task-driven losses.

### Architecture / algorithm
**Flow matching** with the standard linear probability path. Let $x_0$ be a noise latent and $x_1$ a data latent (future frames conditioned on $x_{\text{ctx}}$), with interpolation variable $\tau \in [0,1]$:
$$x_\tau = (1-\tau)x_0 + \tau x_1.$$
For the linear path the ground-truth velocity field is $u^\star(x_\tau, \tau) = x_1 - x_0$ (constant in $\tau$). A network $u_\theta(x_\tau, x_{\text{ctx}}, \tau)$ is trained to approximate it, learning $p(x_{\text{future}} \mid x_{\text{ctx}})$. Sampling integrates the ODE $\frac{dx_\tau}{d\tau} = u_\theta(x_\tau, x_{\text{ctx}}, \tau)$ from $x_0 \sim p_{\text{noise}}$ at $\tau=0$ to $\tau=1$.

**Model.** A Transformer latent generative architecture operating directly in DINOv3 features. Input is the noisy target latent $x_\tau$, conditioned on $x_{\text{ctx}}$ via **cross-attention**. Positional embeddings are decomposed into temporal + spatial components with **RoPE** applied to both $x_\tau$ and $x_{\text{ctx}}$. Two parts: (1) a **backbone** = 2 DiT-style blocks, dimension 256, with query–key normalization for stability; (2) a **projection head** fed via **AdaLN** conditioning — shallow but **wide**: 2 projection layers of dimension **1024**, predicting $u_\theta(x_\tau,\tau)$. The wide head (1024) deliberately exceeds the per-patch latent dim ($d=384$), which the paper finds critical for accurate velocity prediction in high-dim latents (echoing RAE, Zheng et al. 2025).

**Block-level detail (verified against the primary-source PDF/Figure 2 — the arXiv HTML render mis-OCRs the block counts as "22"; the true counts are 2 and 2).** The model is only **4 transformer blocks total**, an unusually shallow design:
- **Backbone block** (dim 256, ×2): `Self-Attention → AdaLN → Cross-Attention (Q=target, K/V=context) → AdaLN → FFN → AdaLN`, each sub-layer residual and gated by AdaLN conditioned on the timestep $\tau$ embedding. Query–key normalization (Henry et al. 2020) stabilizes attention at this feature scale.
- **Wide head block** (dim 1024, ×2): conditioned on the backbone's output *via AdaLN* (not concatenation) — Figure 2 shows the same shape as the backbone (cross-attention-to-context + FFN + AdaLN) just at 4× the width, outputting the final velocity $u_\theta(x_\tau,\tau)$ projected back down to $D=384$.
- RoPE (temporal + spatial decomposed) lets attention resolve "which frame / which patch" without learned absolute position embeddings — this is what makes cross-attention correctly align a target patch with the matching context patches.
- The paper's own ablation (depth 2→8 → no gain; width is the real lever, saturating around 1024) explains *why* they landed on this cheap-backbone / expensive-head shape: DINOv3 already supplies the semantic/dynamics prior, so the flow-matching head only has to learn a comparatively simple velocity field on top of it.
- **Not specified anywhere in the paper**: attention head count, FFN expansion ratio. Code is public (`github.com/facebookresearch/flowwm`) if exact values are needed.

**Parameter count [analyst's view / estimate — not stated in the paper].** Grepping the full extracted PDF text for "param" turns up nothing about model size. Back-of-envelope with standard QKVO attention and an assumed 4× FFN ratio (unconfirmed): backbone ≈ 2M, wide head ≈ 25M → **roughly 30–40M trainable parameters total**, order-of-magnitude only. This sits in front of a separately **frozen** DINOv3 ViT-S encoder (~21M params, untrained) that does the actual visual representation learning. For scale, this is ~3 orders of magnitude smaller than DreamZero's 14B video-diffusion backbone ([[papers/ye-2026-world-action-models]]) — consistent with FlowWM's thesis that the encoder (not the generative head) is where the heavy lifting belongs.

### Training objectives
**(1) Standard flow-matching loss** — conditional expected squared error:
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{x_1 \sim p_{\text{data}},\, x_0 \sim p_{\text{noise}}}\big[\, \| u_\theta(x_\tau, \tau) - (x_1 - x_0) \|_2^2 \,\big].$$

**(2) Temporal consistency via one-step projection.** Imposing losses on the true endpoint $x_1$ normally requires integrating the ODE (expensive, unstable gradients). Instead they get a **differentiable endpoint estimate** by a single linear step along the current velocity:
$$\tilde{x}_1(\tau) = x_\tau + (1-\tau)\, u_\theta(x_\tau, \tau).$$
Then, inspired by Physics-Informed NNs (regularizing derivatives, not just outputs), they match ground-truth temporal derivatives. With $\Delta x_t^{\text{GT}} = x_{t+1}^{\text{GT}} - x_t^{\text{GT}}$:
$$\mathcal{L}_{\text{temporal}} = \sum_{t=1}^{T-1} \big\| \Delta \tilde{x}_t - \Delta x_t^{\text{GT}} \big\|^2,$$
where the $\tilde{x}$ are the projected endpoints. This pushes the *end-of-flow* latents to have the same frame-to-frame dynamics as real latents — and incurs **no extra training compute** (the projection is one add).

**(3) Task-driven objective.** The same one-step projection $\tilde{x}_1$ lets a **frozen** downstream head supervise the world model. E.g. a frozen object detector's loss $\mathcal{L}_{\text{det}}(\tilde{x}_1)$ is backpropagated **only through the projection and the world model** — never through the full denoising trajectory — steering predictions toward semantically useful features. This avoids the expensive/unstable backprop-through-time of reward-based diffusion finetuning (ImageReward, Clark et al. 2024).

### Timestep-schedule shifting
When the generated tensor's dimensionality (or horizon) grows, the training timestep distribution and the inference scheduler must shift toward **noisier** timesteps to keep a stable signal-to-noise ratio. They use the resolution-dependent shift of Esser et al. (2024), rescaling by factor $\alpha$:
$$\tau' = \frac{\alpha\, \tau}{1 + (\alpha - 1)\tau},$$
which preserves the endpoints $\{0,1\}$ while redistributing probability mass to higher noise. Originally for high-res image gen, here justified because raising latent dimension or prediction horizon likewise increases the number of noise dimensions.

### Inference / sampling
Euler ODE solver, **50 steps**, integrating $\tau: 0 \to 1$ from Gaussian noise. Being stochastic, multiple futures are sampled per context; evaluation uses **best-of-$N$** metrics (e.g. $\text{AP}_L(N)$ = max AP over $N$ samples).

## Experimental setup

- **Encoder:** frozen DINOv3 **ViT-S**, $D=384$; each frame encoded separately; average-pool token features from every third layer (3, 6, 9, 12), motivated by intermediate layers giving stronger representations (Bolya et al. 2025).
- **Datasets/benchmarks:** (a) **Bouncing Shapes** — synthetic; a red square + blue ball bounce in a 2D box; at each wall collision the object does a standard bounce *or* reverses velocity with prob 0.5, giving exponentially branching, *enumerable* futures. 16 context → 16 predicted frames. (b) **FuturePerception** — built on the Waymo Open Dataset (recreated because Walker et al. released no code/splits); no filtering; 4 context → 12 predicted frames (1.2 s at 10 FPS); Waymo frames resized to $512\times512$.
- **Baselines:** DINO-WM (deterministic, autoregressive); a stronger deterministic predictor (whole future jointly, same architecture as FlowWM); a VAE-latent stochastic WM using the WAN 2.2 VAE ($D=16$); and a re-implementation of Walker et al. (2025) with DINOv3 features (SOTA stochastic, high-dim).
- **Downstream heads:** object detection via DINO-DETR (detrex/detectron2); depth via a lightweight head trained on DINOv3 latents with Depth-Anything-V2 pseudo-labels.
- **Metrics:** synthetic — Precision Error (are samples valid modes?), Recall Error (are all modes covered?), F1 (harmonic mean), all lower-better; real — $\text{AP}_L(N)$ best-of-N for vehicle/pedestrian/cyclist (large objects), depth RMSE and threshold accuracies $\delta_1,\delta_2,\delta_3$, plus FVD on decoded frames.
- **Training:** AdamW, lr $10^{-3}$, effective batch 128, 52K steps (150 epochs), grad clip 1.0, 10-epoch linear warmup + cosine anneal; **64 V100 GPUs, fp32, ~25 h**.

## Key results

- **Synthetic (Bouncing Shapes), F1 error ↓:** DINO-WM 17.8, deterministic predictor 17.1, Walker et al. 14.4, **FlowWM w/o temporal consistency 4.53**, **FlowWM 4.31**; Oracle (ground-truth latents) 1.00. So FlowWM roughly **3× lower F1 error** than the best prior stochastic baseline, improving both precision *and* recall (more accurate *and* better mode coverage).
- **Real-world (FuturePerception), object detection $\text{AP}_L(3)/\text{AP}_L(6)$ ↑:** DINO-WM 14.5/14.5, deterministic 15.2/15.2, WAN-VAE WM 17.5/18.2, Walker et al. 16.5/17.1, **FlowWM 20.9/21.7**; Oracle 65.1. Note VAE latents beat the *deterministic* DINOv3 models on detection but lose to FlowWM; re-encoding VAE predictions back into DINOv3 space *hurts* (17.5→16.5), confirming the information is lost at prediction time and not recoverable post-hoc.
- **Depth ↓RMSE / ↑δ:** FlowWM best across all metrics (RMSE 0.078, $\delta_1$ 0.723 vs ≤0.57 for baselines; Oracle RMSE 0.043).
- **Pixel FVD ↓:** FlowWM 87.3 vs deterministic predictor 152.4 — a **43% improvement** (though pixel gen is not the focus).
- **Horizon robustness:** deterministic predictors' $\ell_1$ latent error grows fast and $\text{AP}_L$ degrades quickly as the horizon lengthens (they drift off any valid mode); FlowWM stays lower-error / higher-AP over long horizons, and latent error correlates directly with detection degradation.

## Ablations

- **Depth vs width.** Increasing DiT depth 2→8 gives **no** detection gain (depth is not the bottleneck). Projection-head **width** matters a lot: narrow heads badly underperform; gains rise up to width 1024 then saturate — width should be ≥ latent dim.
- **Temporal consistency loss** (Eq. 5): +5.2% relative on FuturePerception (baseline 19.1 → 20.1 $\text{AP}_L(3)$), at no extra training cost.
- **Task-driven detector loss:** +further gain (20.1 → 20.9); smaller than temporal consistency, hypothesized bounded by the frozen detector's limited robustness to noisy latents (weak gradients). Weighting the loss toward **later** timesteps (larger $\gamma$) helps — uniform weighting over-constrains the FM objective; too large $\gamma$ saturates/degrades (task-specialization vs generative-flexibility trade-off).
- **Timestep-schedule shift:** skewing training timesteps toward noisier levels is **crucial** for high-dim / long-horizon.
- **Number of samples $N$:** $\text{AP}_L(N)$ rises monotonically with $N$; deterministic models get zero benefit (same output every time).

## Limitations

The authors defer detailed limitations to appendices (J/K) not in the fetched text, so much is inferred. Honest flags: (1) a large **Oracle gap** remains (FlowWM 20.9 vs Oracle 65.1 $\text{AP}_L$; depth RMSE 0.078 vs 0.043) — real-world future prediction is far from solved. (2) Task-driven gains are **small** and gradient-limited by the frozen detector. (3) Best-of-$N$ metrics reward *coverage* — a model that samples diversely and gets lucky on one of $N$ looks good; single-sample quality is less emphasized. (4) Evaluated with a small backbone (DINOv3 ViT-S, $D=384$) and modest architecture (2 DiT blocks, dim 256); scaling behavior is untested here. (5) FuturePerception is a re-creation of Walker et al.'s unreleased protocol, so cross-paper comparison is approximate. (6) 50-step Euler sampling per future × $N$ samples is not cheap for real-time driving use.

## Why it matters [analyst's view]

The clean, quotable thesis is: **accurate latent future prediction needs both stochasticity and semantically-rich representations, and prior work always sacrificed one.** FlowWM's real technical contribution isn't "flow matching for world models" (that exists) but the *recipe* for making generative modeling stable and useful in a high-dimensional *pretrained* feature space — wide projection head, timestep shifting, and especially the **one-step projection** trick, which is the elegant load-bearing idea: it turns an expensive backprop-through-ODE into a single differentiable add, unlocking both a PINN-style temporal-derivative loss and arbitrary frozen-decoder task losses for essentially free. That pattern (supervise the cheaply-estimated endpoint, not the trajectory) is reusable well beyond world models — any flow/diffusion model that wants endpoint-level auxiliary supervision. For the vault this sits at the intersection of flow matching, world models, and the "generate in DINO/representation space rather than VAE space" trend (RAE, JIT, Orbis, Walker et al.), and it operationalizes the *perception-first* evaluation philosophy (judge world models by downstream detection/depth, not FVD). Code is released (github.com/facebookresearch/flowwm), which raises its practical value.

## Open questions / things to verify

- Does the recipe scale to larger DINOv3 backbones (ViT-L/H, larger $D$) and longer horizons, or is the wide-head prescription $\text{width} \geq D$ enough at scale?
- How much of the win is *flow matching* vs simply *modeling multimodality at all*? The synthetic 14.4→4.31 jump suggests the prior stochastic baseline's *design* (not just determinism) was the problem — worth isolating.
- The task-driven loss is gradient-starved by frozen detectors; would a detector finetuned to tolerate noisy latents, or a smoother surrogate, unlock larger gains?
- Real-time feasibility: 50 Euler steps × $N$ samples per rollout — can step count be distilled down without losing mode coverage?
- Does the one-step endpoint estimate $\tilde{x}_1(\tau)$ bias supervision at low $\tau$ (far from the endpoint)? The timestep-weighting ablation hints yes; a principled schedule would help.

## Connections

- Topic MOCs: [[topics/flow-matching]], [[topics/world-models]], [[topics/generative-models]]
- Author indices: [[authors/francois-porcher]], [[authors/nicolas-carion]], [[authors/karteek-alahari]], [[authors/shizhe-chen]]
- Builds on: flow matching (Lipman et al. 2023; rectified flow, Liu et al. 2022), DiT (Peebles & Xie 2023), timestep shift (Esser et al. 2024), RAE / diffusion-in-DINO-space (Zheng et al. 2025), DINOv3 (Siméoni et al. 2025)
- Contrasts with / extends: DINO-WM (Zhou et al. 2024, deterministic), DINO-Foresight (Karypidis et al. 2024), Walker et al. (2025, closest stochastic-feature prior), Orbis (Mousakhan et al. 2025, distills DINO→VAE)

## Selected quotes

> "Deterministic predictors trained with standard regression losses tend to average over these possibilities, producing predictions that may correspond to no valid future." — §1 (proof in Appendix A)

> "The wide projection dimension (1024) ensures the head's hidden dimension exceeds the per-patch latent dimension (d=384), which is critical for accurately predicting velocity fields in high-dimensional latent spaces." — §3.2

> "FlowWM sharply reduces the F1 error from 14.4 to 4.53 even without temporal consistency. Adding the temporal consistency loss further improves the F1 error to 4.31." — §5.2
