---
type: paper
title: "Back to the Features: DINO as a Foundation for Video World Models"
authors: ["Federico Baldassarre", "Marc Szafraniec", "Basile Terver", "Vasil Khalidov", "Francisco Massa", "Yann LeCun", "Patrick Labatut", "Maximilian Seitzer", "Piotr Bojanowski"]
year: 2025
venue: arXiv preprint
url: https://arxiv.org/abs/2507.19468
rw_id: 01kx5vn8bvvs20ytqwme20ak75
topics: [world-models, representation-learning, self-supervised-learning, jepa]
priority: high
read_state: queued
relevance: ""
added: 2026-07-14
last_updated: 2026-07-14
---

# Back to the Features: DINO as a Foundation for Video World Models

## TL;DR

DINO-world is a generalist video world model that predicts *future frames in the frozen latent space of DINOv2* rather than in pixel space. A DINOv2 ViT-B/14 encoder maps each frame to patch tokens; a large cross-attention transformer "predictor" (up to ViT-g scale, ~1.1B params) is trained by next-frame teacher-forcing on ~60M uncurated web videos to predict the next frame's patch features from all past patch tokens, with a smooth-L1 loss and 3-axial RoPE that lets it handle variable frame rate, resolution, and context length. Because it reuses DINOv2's semantic/geometric features and never renders pixels, it is far cheaper than pixel-space giants (COSMOS used ~22M GPU-hours and up to 12B params; DINO-world's giant predictor trains in ~95.6 hours on 16 H100 nodes). It beats V-JEPA and COSMOS on dense forecasting proxies — e.g. **+6.3 mIoU over the second-best model on VSPW segmentation forecasting at 0.5s** — and matches strong models on intuitive-physics benchmarks. It can then be cheaply *post-trained* with lightweight "action blocks" for planning, where large-scale pre-training beats training from scratch.

## Context & motivation

World models — networks that predict an environment's future state from past observations (and optionally actions), popularized by Ha & Schmidhuber (2018) — have surged back with conditional generative video models on narrow domains (driving: GAIA; games: Genie) and large text-to-video systems (SORA, MovieGen, Wan2.1, COSMOS) reframed as "world simulators." The paper argues three problems block progress: (1) a **data problem** — action-annotated video is scarce, proprietary, and task-specific; (2) the **task is hard and wasteful in pixel space** — pixel generation is resource-hungry (COSMOS: ~22M GPU-hours) and squanders capacity on irrelevant detail (e.g. every leaf in the wind for a driving system); and (3) **evaluation is fragmented** across generation quality, forecasting, physics, and control.

The stated contribution is to **pre-train a video world model in the latent space of a frozen vision encoder (DINOv2)**, then optionally post-train it on actions. The claimed advantages: (1) decoupling unconditional video pre-training from action fine-tuning lets the model learn general dynamics from unlabeled video and reduces the need for action labels; (2) predicting latent features sidesteps per-pixel modeling unnecessary for most downstream tasks; (3) the frozen encoder bootstraps learning by reusing DINO's semantics and avoids the complexity of jointly training encoder + predictor (the paper positions this against V-JEPA [24], whose jointly-trained predictor gives good summarization features but — per their Table 1 — poor forecasting/planning features).

## Method

### Problem formulation
A video is $T$ frames with timestamps: $\{(v_t, \tau_t)\}_{t=1}^{T}$, $v_t \in \mathbb{R}^{H'\times W'\times 3}$, $\tau_t \in \mathbb{R}_+$. Unlike the common fixed-FPS simplification, they **explicitly keep timestamps** $\tau_t$ (in seconds) to support variable FPS and fine-grained temporal control. The frozen encoder gives the **state**: $x_t = \text{ENCODER}(v_t) \in \mathbb{R}^{H\times W\times D}$, with patch token $x_{t,i,j}$ the atomic unit. A world model is the mapping (Eq. 1):

$$(\mathbf{X}_{1:t}, \mathcal{T}_{1:t}, (\tau_{t'}, i', j')) \to \mathbf{x}_{t',i',j'} \quad \forall (i',j'),\ \forall t' > t,$$

i.e. given past features and timestamps and a *query* future coordinate $(\tau_{t'}, i', j')$, predict the patch token there.

### Core idea
Freeze a strong self-supervised image encoder (DINOv2) and spend all the modeling capacity on a large predictor that forecasts future *features*, not pixels — turning world modeling into feature forecasting in a semantic latent space.

### Architecture / algorithm
The predictor is a stack of $N$ residual **pre-norm cross-attention** blocks (inspired by NMT decoders and cross-attention MAE decoders). To predict coordinate $(\tau_{t'}, i', j')$, a **query token** $q \in \mathbb{R}^{D'}$ is initialized from a learnable embedding and, at each block, cross-attends to key-value pairs from all *past* patch tokens, followed by an MLP (Eq. 3):

$$q \leftarrow q + \text{CrossAttn}(\text{LN}(q),\ \{x_{t,i,j} \mid \tau_t < \tau_{t'}\})$$
$$q \leftarrow q + \text{MLP}(\text{LN}(q))$$

After the last block, a LayerNorm + linear projection maps $q$ back to the encoder's $D$-dim space to give $\hat{x}_{t',i',j'} \in \mathbb{R}^{D}$. Encoder features are first linearly projected up to the transformer width $D'$.

**Positional encoding.** Queries and context carry no location info intrinsically, so RoPE is injected into attention. The head dimension is split into **three axes** — temporal $\tau$, horizontal $i$, vertical $j$ — encoded separately (3-axial RoPE). Spatial coordinates use *relative* positions on a $[-1,+1]^2$ grid so changing input resolution does not change relative patch distances; the temporal coordinate uses **absolute timestamps in seconds** so the model distinguishes high vs. low FPS and can extrapolate to longer videos. RoPE angular periods span $\omega = 10^{\text{linspace}(-2,2,\text{steps}=10)}$ (Eq. 5). Concretely (App. A.1) each 60-dim head is split into three 20-dim chunks (10 periods each), with the trailing 4 dims left unrotated.

**Training objective (Eq. 4).** For parallelism, they train with **next-frame prediction** ($t' = t+1$) and **teacher forcing**. Given $T$ frames, all predictions $\hat{x}_{t+1,i',j'}$ for $t \in \{1,\dots,T-1\}$ and every $(i',j')$ are computed *in parallel* by stacking $(T-1)HW$ queries and applying a **block-triangular attention mask** enforcing causality (a query for frame $t+1$ attends only to tokens up to frame $t$):

$$\min_\theta\ \mathcal{L}\big(x_{t+1,i',j'},\ \text{PREDICTOR}_\theta(\mathbf{X}_{1:t}, \mathcal{T}_{1:t}, (\tau_{t+1}, i', j'))\big).$$

$\mathcal{L}$ is the element-wise **smooth-L1 (Huber) loss** with $\beta=0.1$ (Eq. 6): $\tfrac{1}{2\beta}(x-\hat x)^2$ if $|x-\hat x|<\beta$, else $|x-\hat x|-\tfrac{\beta}{2}$. Crucially, the loss is computed over **all** $(T-1)HW$ predicted tokens, unlike masked-reconstruction losses (V-JEPA, DINO-Foresight) that only supervise the small fraction of mask tokens — a denser training signal.

**Variable-FPS sampling.** Taking $T$ contiguous frames skews time-deltas $\Delta\tau = \tau_{t+1}-\tau_t$ toward short intervals, limiting predictive horizon. Instead, for each video they sample $T-1$ deltas **uniformly** from $[\Delta\tau_{\min}, \Delta\tau_{\max}]$, cumulative-sum them from a random start to get timestamps, then decode the nearest actual frame per timestamp. This trains on a roughly uniform distribution of time intervals.

### Derivations / why it works
_No formal derivation; empirical paper._ The design rationale (not a proof) is that a frozen semantic latent space removes the burden of pixel reconstruction, so predictor capacity models dynamics; the ablations (below) supply the supporting evidence.

### Training procedure
Main model: DINOv2 **ViT-B/14 with registers** encoder (last-layer features only, $D=768$, 256 patch tokens per $224\times224$ frame; CLS + registers discarded). Predictor is a **ViT-g-scale cross-attention transformer**: $N=40$ blocks, $D'=1536$, 24 heads, MLP ratio 4.0, ~1.1B params. Optimizer **AdamW**, 300k iterations, batch 1024 clips, $T=8$, resolution $224\times224$, then **50k more iterations at $448\times448$**. LR linearly warmed up to $10^{-4}$ over 5k iters then held **constant**; weight decay constant 0.4 (cosine LR / ramped WD gave no significant difference). Data: a private pool of **~66M uncurated web videos**, 5–60 s, mixed frame rates. Compute (per Table 5): base (86M) = 4 nodes / 45.3 h; large (304M) = 8 nodes / 70.2 h; giant (1.1B) = 16 nodes / 95.6 h, each node 8× H100-80GB.

### Inference / sampling
Because the model accepts arbitrary query timestamps, at test time it can either **directly** query a far-future timestamp, or **autoregressively** roll out intermediate frames (feeding its own predictions back as context in place of encoder features). Predicted features aren't rendered to pixels but are fed directly to task heads (segmentation/depth) or to a planner.

### Action-conditioned fine-tuning (Sec. 3.3)
For observation-action trajectories $(v_t, a_t)$, they add a zero-initialized **action block** after each transformer block that updates the query as $q + \text{MLP}(\text{LN}([q, a]))$ (concatenating query and action embedding), so at init it is the identity. Only the action blocks need training (base model can stay frozen), which curbs overfitting and lets one base model serve many tasks. Adding 12 action blocks to the base predictor adds ~22M params. This is contrasted with **DINO-WM's** approach of interleaving action tokens into the patch sequence, which complicates batching/masking and forces full fine-tuning (risking destroying learned video understanding).

## Experimental setup

- **Dense forecasting** (proxy for feature quality): train a "present-time" linear head (segmentation / depth) on DINOv2 features, then apply it to features predicted ~200 ms (short) or ~0.5 s (mid) ahead, given 4 past frames. Datasets: **Cityscapes, VSPW (mIoU ↑), KITTI (RMSE ↓)**. For COSMOS, generated pixels are re-encoded with DINOv2 and the same heads applied.
- **Intuitive physics**: **IntPhys, GRASP, InfLevel** using Garrido et al.'s protocol — a "surprise" score (mean absolute error between predicted and actual features; perplexity for COSMOS's discrete tokens) should be higher for physically implausible videos.
- **Baselines**: DINO-Foresight (masked-reconstruction, Cityscapes-only), V-JEPA (ViT-L/H, joint encoder+predictor), COSMOS-4B/12B (pixel-space, ~100M curated videos), and a "Copy Last" baseline (last observed frame's features as the prediction).
- **Planning**: three simulated RL environments **PushT, Wall, PointMaze** (setup of Zhou et al. / DINO-WM), with Cross-Entropy Method planning over latent rollouts, success over 512 episodes/env.

## Key results

- **Dense forecasting (Table 1).** DINO-world (ViT-B encoder) has present/short/mid VSPW mIoU of **52.8 / 51.6 / 47.0** and Cityscapes **68.6 / 64.7 / 55.1**, with small present→forecast gaps (a strong-world-model signal). It beats COSMOS-12B (VSPW mid 40.7; Cityscapes mid 45.9) and dramatically beats V-JEPA (VSPW mid collapses to ~4–8 mIoU). The abstract/intro headline: **+6.3 mIoU over the second-best model on VSPW mid-term (0.5 s) forecasting.** DINO-Foresight is marginally better only on its home domain (Cityscapes, KITTI), attributed to domain-specific driving training.
- **Intuitive physics (Table 2).** DINO-world (ViT-B enc / 1.1B predictor) scores **IntPhys 91.3, GRASP 76.0, InfLevel 63.7** — comparable to V-JEPA ViT-H (89.4 / 73.0 / 59.9) with a smaller encoder. COSMOS-4B is near-perfect on the simple IntPhys (99.5) but weaker on GRASP/InfLevel (60.1 / 44.8). The authors treat these as a *sanity check, not a benchmark*, given distribution shift and noise.
- **Planning (Table 4).** Per-environment success rates are reported in the paper's table (not fully transcribed here — _needs verification_ for exact per-env numbers), but the stated finding is clear: **large-scale pre-training improves planning success over training the predictor from scratch** on PushT, Wall, and PointMaze, with the benefit expected to grow on more complex environments closer to the pre-training distribution.
- **Efficiency.** Effective world models with **<1B params** vs. COSMOS's up to 12B, and vastly fewer GPU-hours.

## Ablations

- **Predictor size (Table 3 left):** clear scaling — base (86M) → large (304M) → giant (1.1B) monotonically improves IntPhys (84.9 → 89.1 → 90.6), Cityscapes-mid (47.7 → 51.9 → 53.2), VSPW-mid (45.4 → 46.4 → 46.8). Suggests temporal-dynamics modeling needs *more* capacity than static spatial features.
- **Training data (Table 3 middle):** the small/narrow Cityscapes and SSv2 datasets do poorly vs. the ~66M web-video collection (e.g. VSPW 23.1 / 45.2 vs. 46.8) — large-scale, diverse data is essential for a generalist model.
- **Visual encoder (Table 3 right):** DINOv2 beats SigLIP2 (slightly worse, attributed to noisy vision-language features) and crushes the SD3.5 VAE (IntPhys collapses to ~13.0 / CS 1.5), confirming a representation-learning encoder is the right substrate — a reconstruction VAE would need a COSMOS-scale predictor.
- **Direct vs. autoregressive (Fig. 3):** direct queries win at short horizons; autoregressive rollouts hold up better at longer horizons; **all** methods degrade sharply as the interval approaches 1 s.

## Limitations

- **Long-horizon prediction fails** — accuracy collapses beyond ~1 s; predictions grow blurry as horizon and uncertainty increase (the model predicts a mean-like future rather than sampling one plausible future).
- **No pixel decoding** — outputs are features, usable for dense prediction/planning but not directly renderable video.
- Intuitive-physics benchmarks are noisy / distribution-shifted (authors' own caveat).
- Planning validated only in **simulation**; real-world post-training/planning is future work.
- [analyst's view] Main pre-training data is a **private ~66M-video pool**, limiting reproducibility; ablation open-data models are much weaker.

## Why it matters [analyst's view]

This is a clean, resource-conscious realization of LeCun's JEPA/world-model program: predict in an abstract latent space, decouple perception from dynamics, and post-train for control. It reframes "world model" away from expensive pixel generation (SORA/COSMOS) toward *feature forecasting on a frozen SSL backbone*, and empirically argues that pixel fidelity is largely wasted compute for control and forecasting. The action-block design is an appealingly modular adapter story — one base dynamics model, cheap per-task heads — echoing the pre-train/adapt split that made LLMs practical. For the vault it sits directly alongside the JEPA and feature-space world-model cluster ([[papers/maes-2026-leworldmodel]], [[papers/porcher-2026-flowwm]], [[papers/kerssies-2026-delta-tokens]]) and the intuitive-physics-in-video thread ([[papers/joseph-2026-physics-video-world-models]], [[papers/zhao-2026-phyworld]]). It is essentially the "scaled-up, unconditional-pretraining" successor to DINO-WM (Zhou et al.), which the vault does not yet hold.

## Open questions / things to verify

- Exact per-environment planning success rates in Table 4 (frozen-action-blocks vs. full-finetune vs. from-scratch) — _needs verification_.
- How well does the smooth-L1 mean-seeking objective interact with the acknowledged long-horizon blur? Would a sampling/stochastic future head (mentioned as future work) fix the collapse past 1 s?
- Does the frozen-encoder choice cap the ceiling — would a lightly-tuned encoder help hard dynamics the DINOv2 features don't capture?
- Real-world (non-simulated) planning and language conditioning are promised but untested.

## Connections

- Builds on: DINOv2 (Oquab et al., frozen encoder) — _needs note_; DINO-WM (Zhou, Pan, LeCun, Pinto — action-conditioned planning on frozen features) — _needs note_; V-JEPA (Bardes et al.) — _needs note_; DINO-Foresight (Karypidis et al.) — _needs note_.
- Contrasts with: pixel-space generative world models — COSMOS, SORA, Wan2.1 — _needs note_.
- Related in vault: [[papers/maes-2026-leworldmodel]], [[papers/porcher-2026-flowwm]], [[papers/kerssies-2026-delta-tokens]], [[papers/joseph-2026-physics-video-world-models]], [[papers/daithankar-2026-temporal-difference-vision]], [[papers/ye-2026-world-action-models]], [[papers/zhao-2026-phyworld]], [[papers/higuera-2026-visuo-tactile-world-models]], [[papers/ding-2024-diffusion-world-model]]
- Topic MOCs: [[topics/world-models]], [[topics/representation-learning]], [[topics/self-supervised-learning]], [[topics/jepa]]
- Author indices: [[authors/yann-lecun]]

## Selected quotes

> "We present DINO-world, a powerful generalist video world model trained to predict future frames in the latent space of DINOv2." — Abstract

> "predicting the exact motion of each leaf in the wind for a system designed for autonomous driving. Capturing the environment at an appropriate level of detail becomes therefore crucial for the effectiveness and efficiency of the model." — §1

> "V-JEPA jointly trains an encoder and a predictor at scale, obtaining features suitable for video summarization, but suboptimal for forecasting and planning, as shown in Table 1." — §2.2

> "all predictions become inaccurate as the forecasting interval approaches 1 second. Forecasting at longer horizons remains a limitation of current models." — §4.2
