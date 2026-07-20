---
type: paper
title: "A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens"
authors: ["Tommie Kerssies", "Gabriele Berton", "Ju He", "Qihang Yu", "Wufei Ma", "Daan de Geus", "Gijs Dubbelman", "Liang-Chieh Chen"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2604.04913
rw_id: 01kx5r751ctjc63kppjkw332ne
topics: [world-models, video-generation, generative-models, representation-learning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-10
last_updated: 2026-07-10
---

# A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens

## TL;DR

The paper introduces **DeltaTok**, a tokenizer that compresses the *change* between two consecutive frames — measured in the feature space of a frozen vision foundation model (VFM, here DINOv3) — into a *single continuous "delta token"*, and **DeltaWorld**, a generative world model that operates entirely on 1-D sequences of these tokens. The core trick is to combine delta tokenization with a **Best-of-Many (BoM)** training objective: at each step the predictor consumes K Gaussian noise queries, produces K candidate futures in one forward pass, and backpropagates only through the candidate closest to ground truth. This collapses video from a 3-D spatio-temporal tensor to a 1-D temporal sequence (e.g. a claimed 1,024× token reduction at 512×512), making it cheap to sample many diverse futures without iterative denoising. On the DINO-world dense-forecasting benchmark (segmentation on VSPW/Cityscapes, depth on KITTI), DeltaWorld's best-of-20 predictions beat the generative Cosmos baseline on essentially all metrics while its mean predictions recover to the discriminative baseline. Headline efficiency: **over 35× fewer parameters and ~2,000× fewer FLOPs than Cosmos** (3.1×10⁴ vs 6.0–6.4×10⁷ GFLOPs to generate 20 samples). The delta formulation also carries a natural prior — predicting "no change" preserves the previous frame — which the authors credit for recovering mean-prediction quality.

## Context & motivation

A world model predicts future states of an environment to let an agent plan. Because the future is uncertain, a good world model must produce a *set* of plausible futures. **Discriminative** world models produce a single deterministic prediction that, under uncertainty, collapses toward the conditional mean and cannot represent distinct future events (the paper cites this regression-to-the-mean pathology, ref [70]). This motivates a **generative** world model.

However, existing generative world models are computationally inefficient for three stated reasons: (i) their representation space is optimized for pixel-level fidelity rather than semantic understanding; (ii) they need multiple sequential forward passes (autoregressive decoding or diffusion denoising) to produce even a single future; and (iii) they fail to exploit the spatio-temporal redundancy of consecutive frames (static backgrounds, only small local changes). Recent work has shifted prediction into the **feature space of vision foundation models** (VFMs) — e.g. DINO features — which improves dense-forecasting accuracy with far fewer world-model parameters than pixel-reconstruction approaches; but most such feature-space models remain *discriminative*.

The paper builds directly on **DINO-world** [3] (Baldassarre et al., "Back to the Features: DINO as a Foundation for Video World Models", ICML 2025), a discriminative VFM-feature-space predictor, and makes it generative and efficient by (a) the BoM objective and (b) delta tokenization. The stated contributions are exactly these two: compressing frame differences into single delta tokens, and building a compact single-forward-pass generative world model on them.

## Method

### Problem formulation

Given a sequence of $t$ video frames $V_{1:t} = (v_1,\dots,v_t)$ with $v_i \in \mathbb{R}^{H'\times W'\times 3}$, a **frozen VFM** $\phi$ embeds each frame into a grid of patch tokens:
$$x_i = \phi(v_i) \in \mathbb{R}^{H\times W\times D},$$
where $x_{i,h,w}\in\mathbb{R}^{D}$ is the patch token at spatial position $(h,w)$. The encoded context is $X_{1:t}=(x_1,\dots,x_t)$ with timestamps $T_{1:t}=(\tau_1,\dots,\tau_t)$. The task is to predict the VFM features of a future frame at target timestamp $\tau_{t+1}$, conditioned on the context. Operating in feature space (not pixel space) abstracts away pixel-level variability so a compact predictor can capture the dynamics.

### Core idea

Two ideas stacked: (1) map different stochastic noise queries to different plausible futures in a *single* forward pass via a Best-of-Many objective, replacing iterative diffusion sampling; (2) since consecutive frames differ only in structured, low-dimensional ways, encode the *difference* between two frames' VFM features into a **single** delta token, so the world model predicts one token per frame instead of an $H\times W$ grid.

### Architecture / algorithm

**Discriminative base predictor (from DINO-world).** The future predictor $f$ forecasts each patch token $\hat{x}_{t+1,h,w}$ using a stack of Transformer blocks that cross-attend from a single learnable query embedding $q$ to the context:
$$\hat{x}_{t+1,h,w} = f(q, X_{1:t}, T_{1:t}, \tau_{t+1}, h, w) \in \mathbb{R}^{D}. \tag{1}$$
This runs independently per spatial location $(h,w)$; positional embeddings make predictions position-dependent. Training uses **teacher forcing** with a **causal attention mask** and a **smooth L1 loss** $\ell$ (with $\beta=0.1$) between predicted and ground-truth features, so all timestamps and context lengths are predicted in parallel in one forward pass. At inference it can roll out autoregressively by appending $\hat{x}_{t+1}$ to the context.

**Best-of-Many (BoM) training (Section 3.2).** To make it generative, the single learned query is replaced by $K$ noise queries drawn from a Gaussian and **shared across all spatial locations**:
$$q^k \sim \mathcal{N}(\mu,\Sigma), \quad k=1,\dots,K. \tag{2}$$
Each yields a prediction:
$$\hat{x}^k_{t+1,h,w} = f(q^k, X_{1:t}, T_{1:t}, \tau_{t+1}, h, w) \in \mathbb{R}^{D}. \tag{3}$$
Only the single best candidate (summed over all spatial locations) is supervised:
$$k^* = \arg\min_k \sum_{h,w}\ell\!\left(x_{t+1,h,w}, \hat{x}^k_{t+1,h,w}\right); \qquad
\mathcal{L}_{\text{BoM}} = \sum_{h,w}\ell\!\left(x_{t+1,h,w}, \hat{x}^{k^*}_{t+1,h,w}\right). \tag{4}$$
This "winner-take-all over samples" pushes different noise queries to cover different plausible futures, while preserving single-pass efficiency. In practice the candidate-selection pass uses **detached parameters** (no activation storage), and only the winner $k^*$ is re-run with gradients — so K=16 costs roughly the same memory as the deterministic baseline.

**Frame compression to a single token (Section 3.3) — the intermediate step.** Predicting/evaluating $K$ full $H\times W$ feature maps per future is expensive, so a continuous autoencoder tokenizer is introduced. Encoder $g$ compresses a feature map $x_t$ plus a learnable embedding $z_{\text{init}}$ into one **frame token**:
$$z_t = g(x_t, z_{\text{init}}) \in \mathbb{R}^{D}. \tag{5}$$
Decoder $h$ reconstructs the map from $z_t$ using $H\times W$ zero-initialized patch tokens $x^{\text{init}}$:
$$\hat{x}_t = h(x^{\text{init}}, z_t). \tag{6}$$
Both are Transformer stacks with self-attention, trained separately (before the world model) with a reconstruction loss:
$$\mathcal{L}_{\text{tok}} = \|x_t - \hat{x}_t\|^2. \tag{7}$$
The BoM loss can then be computed *directly in single-token space*, avoiding decoding during predictor training. Limitation: one token must represent the entire scene at each timestep, which caps fidelity and hurts prediction accuracy (confirmed empirically — see step (2) in Ablations).

**Delta compression to a single token — DeltaTok (Section 3.4).** Instead of re-encoding the whole frame, condition the tokenizer on the *previous* frame so the token only needs to encode *how to transform* $x_{t-1}$ into $x_t$ — much less information. The encoder now takes both frames:
$$z_t = g(x_{t-1}, x_t, z_{\text{init}}) \in \mathbb{R}^{D}, \tag{8}$$
and the decoder reconstructs the current features by transforming the previous ones with the delta token:
$$\hat{x}_t = h(x_{t-1}, z_t). \tag{9}$$
DeltaTok reuses the frame-compression architecture and the same MSE reconstruction loss (Eq. 7), with frame pairs $(x_{t-1}, x_t)$ drawn from the same uniform timestamp-sampling used for the predictor. A single delta token can span the range from near-static scenes (retain most of the previous frame) to large transitions (encode a near-absolute new state); the **inference frame rate controls how much change each token represents**. Because DeltaTok is non-spatial and semantic, it naturally handles occlusions and new objects where warping/optical-flow residuals struggle, and can "revert to absolute compression" when temporal redundancy is low.
![[Pasted image 20260710134247.png]]

**DeltaWorld — putting it together.** Combine a separately trained, **frozen** DeltaTok with the predictor $f$. The input sequence is **prepended with a black frame** so the first delta token $z_1$ effectively encodes the *absolute* features of the first real frame. The predictor operates on the past delta-token sequence $Z_{1:t}=(z_1,\dots,z_t)$ and predicts the next delta token:
$$\hat{z}_{t+1} = f(q^k, Z_{1:t}, T_{1:t}, \tau_{t+1}). \tag{10}$$
The spatial feature map is recovered on demand via the DeltaTok decoder: $\hat{x}_{t+1} = h(x_t, \hat{z}_{t+1})$. During training the BoM winner is chosen **in delta-token space, without decoding**. At inference, different noise queries yield diverse futures in one forward pass; autoregressive rollout appends each predicted delta token to the context, staying entirely in delta-token space, and the decoder is applied separately to recover spatial features for downstream heads.

Note the architectural simplifications enabled by one-token-per-frame: the DeltaWorld predictor drops the 3D-RoPE and block-causal mask of DINO-world in favour of a **1D RoPE** (rotating the first 60 of each head's dims, last 4 unrotated) and a **standard causal (diagonal) mask**. Noise queries are sampled from $\mathcal{N}(0, 0.02^2 I)$.

### Derivations / why it works

_No formal derivation; the paper is empirical/methodological._ The load-bearing arguments are conceptual rather than proven:

- **Why BoM yields diversity without denoising.** The winner-take-all loss (Eq. 4) supervises only the closest candidate, so gradient never pushes all queries toward the same conditional mean. Different $q^k$ are therefore free to specialize to different modes — a single-pass alternative to diffusion's iterative denoising. The paper is explicit (Appendix D) that, unlike diffusion, BoM has *no explicit distributional objective*: coverage is bounded by K and there is no guarantee the sampled-future frequencies approximate true outcome probabilities.
- **Why delta tokens recover mean quality.** The delta formulation has a "natural prior": predicting *no change* simply preserves the previous frame (Eq. 9 with a near-identity delta). This makes the easy-but-common outcome cheap to represent and, the authors argue, is why DeltaWorld's mean-of-20 recovers to the discriminative baseline (44.4 vs 44.8 VSPW; 45.5 vs 45.4 Cityscapes) whereas naive BoM collapsed.
- **Why one token suffices for a delta but not a frame.** A frame token must carry the whole scene (limited capacity → low fidelity); a delta token only carries the low-dimensional structured change between consecutive frames, so a single continuous vector represents it far more faithfully.

### Training procedure

- **VFM backbone:** frozen **DINOv3** [60], **ViT-B** variant. Tokenizer and predictors also use ViT-B for simplicity (formulation is scale-agnostic).
- **Training data:** ~4M video samples spanning diverse domains, mostly 640×360, mean duration ~11 s, 16 FPS (Table A). This mirrors the DINO-world corpus (which is ~66M samples, unreleased). None of the evaluation datasets are in training.
- **Tokenizer:** trained *separately first*, 50K iterations at each resolution, MSE loss, **AdamW**, linear warmup to lr $10^{-3}$ over 5K steps then constant, weight decay $10^{-4}$, batch size 1,024, gradient-norm clip at $10^{-2}$. Encoder/decoder are ViTs with 2D RoPE; encoder adds a learned per-frame embedding to distinguish previous- vs current-frame tokens; Layer Scale initialized to $10^{-5}$ and final decoder LayerNorm omitted so the decoder is ~identity at init.
- **Predictor:** **AdamW**, lr $10^{-4}$ with 5K-step linear warmup then constant, weight decay $4\times10^{-1}$, smooth L1 loss ($\beta=0.1$), batch size 1,024, training sequence length 8 frames, **no gradient clipping**. Main results: **300K iterations at 512×512 with K=256**. Ablations: 100K iterations at 256×256 (K=16 in Table 2). Predictors are then fine-tuned 5K iterations at 10× lower learning rate.
- **Augmentation:** random resized crops (scale 0.6–1.0, aspect 3:4–4:3) applied consistently across the sequence then squared. Temporal offsets $\Delta\tau$ sampled uniformly from $[1/25, 1/3]$ s (enables arbitrary future timestamps).
- **Compute note:** training time/memory measured on a single node of 8× NVIDIA H200, BF16 mixed precision, `torch.compile`.

### Inference / sampling

Following the benchmark protocol: a **four-frame context**, **direct prediction** for short-term (~0.2 s), and **three-step autoregressive rollout** for mid-term (~0.6 s). For generative models, **K futures are rolled out independently**, each sampling a fresh noise query at every step and appending its own prediction to its own context. At test time **20 samples** are drawn; `best` = sample with lowest DINOv3 feature-space loss to ground truth at the last predicted step; `mean` = average the 20 DINOv3 feature maps at the last step, then apply the task head once (feature-space averaging enables fair comparison with a single-prediction discriminative model). Linear segmentation/depth heads are trained on frozen VFM features and then applied to the predicted future features. For the pixel-generating Cosmos baseline, predicted pixels are re-encoded with the same VFM for feature-level comparability.

## Experimental setup

- **Benchmark:** the DINO-world dense-forecasting benchmark [3], on **unseen** evaluation datasets.
- **Datasets (Table 1):** VSPW (segmentation, various domains, 343 sequences), Cityscapes (segmentation, driving, 2048×1024, 500 seq), KITTI (monocular depth, driving, 1216×352, 28 seq). Horizons: short ~0.2 s, mid ~0.6 s.
- **Metrics:** segmentation **mIoU** (higher better), depth **RMSE** (lower better). GFLOPs measured with the DeepSpeed FLOPs Profiler.
- **Baselines:** discriminative **DINO-world** [3] (reimplemented, trained on the same data); generative **Cosmos** [1] at 4B and 12B (latent space trained for pixel reconstruction; both use a separate ~7B diffusion decoder that dominates FLOPs). Bounds: **Copy last** (repeat last observed features) as lower bound; **Present** (ground-truth future features) as upper bound.

## Key results

**Dense forecasting (Table 3, 512×512, best-of-20 with mean in parentheses).** DeltaWorld generally beats both Cosmos sizes on best across all metrics and on mean across nearly all, at ~2,000× fewer FLOPs (3.1×10⁴ GFLOPs vs Cosmos 6.0–6.4×10⁷). Selected numbers:
- **VSPW mIoU:** DeltaWorld 55.4 short / 50.1 mid (best); DINO-world 54.0 / 47.9; Cosmos-12B 51.7 / 47.7. Upper bound (Present) 58.4.
- **Cityscapes mIoU:** DeltaWorld 65.8 short / 55.4 mid (best), and even its mean (63.9 / 51.3) exceeds DINO-world's 62.0 / 49.8. Cosmos-12B 55.3 / 53.3.
- **KITTI RMSE (lower better):** DeltaWorld 3.00 short / 3.88 mid (best); DINO-world 3.16 / 4.07; Cosmos-12B 3.72 / 4.01.
- The best–mean gap is consistently larger for DeltaWorld than Cosmos, which the authors read as *more meaningful sample diversity* (Cosmos's best and mean are nearly identical). Qualitatively (Fig. 6), samples differ in a pedestrian's position and ego-camera motion.

Versus the discriminative DINO-world, DeltaWorld's mean is modestly better on Cityscapes and modestly worse on VSPW/KITTI, while its best substantially exceeds DINO-world's single deterministic prediction — i.e. the sampled futures cover realistic modes a deterministic model cannot.

**Efficiency headline:** >35× fewer parameters and ~2,000× fewer FLOPs than existing generative world models (Cosmos). In DeltaWorld the predictor is only **~0.5% of total inference FLOPs** when generating 20 samples; per-sample cost is dominated by the DeltaTok decoder (~46 GFLOPs/step), while the backbone (188.74) and DeltaTok encoder (387.72) are computed **once and shared across all K samples** (Table B).

## Ablations

**Towards an efficient generative world model (Table 2, mid-horizon mIoU, 256×256, K=16 train, best-of-20 eval; mean in parentheses):**
- **(0) Discriminative baseline (DINO-world reimpl.):** VSPW 44.8, Cityscapes 45.4; 959 GFLOPs (single prediction).
- **(1) BoM training:** best rises to VSPW 47.0 / Cityscapes 46.8, but **mean collapses** (39.4 / 31.1) — many samples degenerate to a single semantic class filling the frame — and training time grows ~5×; predictor is 97% of inference FLOPs (12,013 GFLOPs for 20 samples).
- **(2) Frame compression:** BoM sampling becomes >10× faster than (1) and 5× less memory (single-token context/predictions; BoM loss in token space). Mean improves over (1) (the decoder resists collapse) but accuracy stays below the discriminative baseline (VSPW 45.7/40.3, Cityscapes 42.7/35.5); 6,315 GFLOPs.
- **(3) Delta compression (DeltaWorld):** best improves over (2) and matches/exceeds uncompressed BoM (1) (+1.9 mIoU Cityscapes, within 0.2 VSPW); **mean recovers to the discriminative baseline** (VSPW 44.4 vs 44.8; Cityscapes 45.5 vs 45.4). Predictor is just 0.5% of inference FLOPs; 6,721 GFLOPs, 0.2× memory.

**BoM sample scaling (Fig. 5, Cityscapes mid-horizon).** Increasing training K improves `best` for any fixed >1 eval queries with **no sign of saturation** (model keeps learning more specific futures); K modestly lowers `mean` but it **stabilizes beyond K=64** — added diversity does not cost average quality.

**Delta tokens in other discriminative architectures (Appendix C).** Replacing per-frame patch tokens with a single delta token in the *discriminative* DINO-world performs comparably (−0.2 VSPW, +1.5 Cityscapes) at 0.5× training time and 0.2× memory (Table C). Integrated into **DINO-Foresight** [35] on Cityscapes, delta tokens match the original while using **2048× fewer tokens** (5 vs 10,240; Table D), and even allow replacing factorized space-time attention with standard self-attention and skipping high-res fine-tuning — evidence the representation transfers across architectures.

## Limitations

**Paper's own (Appendix D):**
- **No explicit distribution modeling.** BoM maps noise queries to distinct futures but has no principled connection to the data distribution (unlike diffusion). Coverage is bounded by K, nothing encourages diverse query-space utilization, and sampled-future frequencies need not approximate true outcome probabilities. Upside they flag: distinct queries behaving like *implicit action conditioning*, hinting at future explicit action-conditional generation.
- **Error accumulation.** Reconstructing absolute features requires repeatedly decoding delta tokens conditioned on previous features, so reconstruction errors may compound and cause feature drift; the predictor adds another error source that can compound over multi-step rollouts. Suggested mitigation: have the tokenizer operate on its own reconstructions (sequential rather than parallel-from-ground-truth).

**Honest-reader additions [analyst]:**
- Evaluation is confined to *dense-forecasting proxy metrics* (seg mIoU, depth RMSE) at short/mid horizons (≤~0.6 s, 3-step rollout); no long-horizon or truly interactive/action-conditioned evaluation, and no downstream planning/control task despite the world-model framing.
- The DINO-world baseline is the authors' *reimplementation* (no official code), so absolute baseline numbers depend on faithful reproduction.
- The "best-of-20" headline depends on an oracle selecting the closest sample to ground truth — informative for coverage, but not usable at deployment when the ground-truth future is unknown; `mean` is the deployable proxy and it only matches (not beats) the discriminative baseline.

## Why it matters [analyst's view]

This is a clean efficiency argument that decouples *diversity* from *expensive iterative sampling*: BoM shows you can get multi-modal futures in a single forward pass, and delta tokenization shows that once you predict in a semantic feature space, temporal change is low-dimensional enough to live in one token per frame. The 2,000× FLOP reduction versus Cosmos is the kind of number that makes generative world models plausible as an inner loop for planning/MPC, where you want to roll out *many* candidate futures cheaply. The framing "a frame is worth one token" is a strong compression prior for video: if it holds up at scale it reframes video world modeling as a 1-D sequence problem, which would let language-model-style scaling (context length, predictor size, rollout depth) apply directly — the authors explicitly tee this up.

Two things to watch downstream: (1) the "implicit action conditioning" hint — if noise queries can be steered, this becomes an action-conditional world model without a separate action head; (2) whether the no-distributional-objective weakness bites on tasks that need calibrated probabilities (e.g. collision risk), where a diffusion or flow-matching head over delta tokens might be the natural hybrid. It sits interestingly against the physics-faithfulness line of work in the vault (PhyWorld, PhyGround): DeltaWorld optimizes for *semantic* future alignment in VFM space, not pixel or physical fidelity, so it is a complementary axis rather than a competitor.

## Open questions / things to verify

- How does the 1,024× / 2,000× headline hold at higher resolution, longer context, or larger predictors? The paper only tests ViT-B and ≤6-frame contexts.
- Does feature drift from delta decoding actually degrade long rollouts, and does the "decode-from-own-reconstruction" fix help? Untested here.
- How well-calibrated is the future distribution — does best-of-K improvement translate to any *deployable* selection criterion when ground truth is unknown?
- Can the noise-query space be made into explicit, controllable action conditioning across scenes (the paper's own conjecture)?
- Sensitivity to VFM choice: everything is in DINOv3 space — would a different VFM (CLIP, a pixel-decodable RAE) change the delta-token compressibility?
- Reproducibility of the DINO-world reimplementation vs the original ICML numbers.

## Connections

- Builds on: DINO-world (Baldassarre et al. 2025, VFM-feature-space discriminative video world model) — _no vault note yet_; DeltaWorld is DINO-world + BoM + delta tokens.
- Extends / relates to (vault neighbours, world models): [[papers/ye-2026-world-action-models]], [[papers/joseph-2026-physics-video-world-models]], [[papers/porcher-2026-flowwm]], [[papers/ding-2024-diffusion-world-model]], [[papers/maes-2026-leworldmodel]]
- Sibling triage batch (physics-faithful video world modeling / benchmarks — complementary fidelity axis): [[papers/zhao-2026-phyworld]], [[papers/lin-2026-phyground]]
- Baseline compared against: Cosmos (pixel-latent diffusion world model, ref [1]) — _no vault note_
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/generative-models]], [[topics/representation-learning]]
- Author indices: [[authors/tommie-kerssies]], [[authors/gabriele-berton]], [[authors/ju-he]], [[authors/qihang-yu]], [[authors/wufei-ma]], [[authors/daan-de-geus]], [[authors/gijs-dubbelman]], [[authors/liang-chieh-chen]]

## Selected quotes

> "a single delta token per frame is sufficient to represent consecutive-frame dynamics in VFM feature space, collapsing video from a three-dimensional spatio-temporal representation to a one-dimensional temporal sequence." — §1 Introduction

> "We attribute the recovered mean to a natural prior of the delta formulation: predicting no change simply preserves the previous frame." — §4.4, Step (3)

> "unlike diffusion models, whose denoising objective provides a principled connection to the data distribution, BoM lacks an explicit distributional objective. Consequently, the model's coverage of the predictive distribution is limited by the number of noise queries K explored during training." — Appendix D, Limitations

> "By demonstrating that videos can be represented using only the temporal dimension, delta tokens offer a compact representation for video understanding and generation at scale." — §5 Conclusion
