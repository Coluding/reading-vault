---
type: paper
title: "You Don't Need Strong Assumptions: Visual Representation Learning via Temporal Differences"
authors: ["Ninad Daithankar", "Alexi Gladstone", "Yann LeCun", "Heng Ji"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2606.15956
rw_id: 01kvq3vn2e44tgtc7e8ny4jztj
topics: [representation-learning, self-supervised-learning, temporal-difference-learning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

TDV (Temporal Difference in Vision) is a self-supervised recipe for learning image representations from video that deliberately drops the strong inductive biases modern SSL depends on — augmentations, masking, cropping, contrastive negatives — and replaces them with a single weak, "exactly correct" assumption: **causality**, i.e. the immediate future is predictable from the past. It jointly trains a frame encoder $f_\theta$ and a motion encoder $m_\phi$ so that the current frame's embedding plus an encoded motion vector equals the next frame's embedding: $\hat{z}_{t+1} = z_t + \Delta z_t$. Trained on Something-SomethingV2 with a DINO-style EMA teacher to prevent collapse, TDV matches DINO and iBOT on dense spatial tasks and **beats** them on optical flow (e.g. 9.84 vs 11.31/13.03 EPE-clean on MPI-Sintel, ViT-S) and on stereo-depth bad-pixel rates, while trailing slightly on semantic segmentation. The paper's framing contribution is a controlled experiment showing the optimal strength of inductive bias *decreases* as data scale grows, motivating the whole approach.

## Context & motivation

The paper opens with a "bitter lesson" style argument: across the history of visual representation learning, methods with *weaker* inductive biases have steadily displaced stronger ones — supervised CNNs → contrastive SSL (SimCLR, MoCo) → self-distillation (BYOL, DINO) → modern ViT self-distillation (DINOv2/v3, V-JEPA). Yet even state-of-the-art SSL still leans on image-level biases: augmentations, masking, cropping, or contrastive negative-pair assumptions. The authors argue these encode beliefs that are only *approximately* correct (e.g. "augmented views should be invariant," "random images should be dissimilar"), and at scale these approximations become bottlenecks because they force the encoder to discard factors of variation.

The stated problem: naively removing these biases leaves no learning signal and collapses the representation (they demonstrate this — see Table 1). So the question becomes "what is the weakest assumption that still provides sufficient signal to avoid collapse?" Their answer is **causality**: causes precede effects, so the immediate future is predictable from the past. They argue causality is domain-agnostic and *exactly* rather than approximately correct — next-frame prediction imposes no invariance constraint, so it never requires the encoder to throw away information. Because causality is inherently temporal, this pushes learning onto **video** rather than static images, departing from the standard practice of training image encoders on image datasets. They draw the analogy to Temporal Difference learning in RL (Sutton 1988), hence the name.

## Method

### Problem formulation
Input: consecutive video frames $x_t, x_{t+1}$. A frame encoder $f_\theta$ maps a frame to token embeddings, and a motion encoder $m_\phi$ maps the raw RGB frame-delta to a latent shift. Objective: predict the next frame's representation from the current frame's representation plus the encoded motion, in latent space (a JEPA-style latent prediction, not pixel reconstruction).

### Core idea
The representation of a frame, combined with the *change* between frames, should yield the representation of the next frame — and the change is modeled explicitly (by the motion encoder) rather than discarded via an invariance objective. Because consecutive frames are close in time, the RGB delta is intrinsically low-rank (background is static; only moving regions carry signal), so the motion encoder is pushed to capture compact abstract motion.

### Architecture / algorithm
**Frame encoder.** A ViT $f_\theta$ maps frame $x_t$ to token embeddings:
$$z_t = f_\theta(x_t) \in \mathbb{R}^{n \times D}, \tag{1}$$
where $n$ = number of spatial patches + a [CLS] token, and $D$ is embedding dim.

**Motion encoder.** The raw RGB difference $\Delta x_t = x_{t+1} - x_t$ captures pixel-space change but must be mapped to a latent shift $\Delta z_t$. Since the same pixel change can mean different things in different contexts, the motion encoder is **conditioned on the current frame's embedding $z_t$ via cross-attention**:
$$\Delta z_t = m_\phi(\Delta x_t;\, z_t). \tag{2}$$

**Additive latent composition.** The next-frame prediction is a simple sum:
$$\hat{z}_{t+1} = z_t + \Delta z_t. \tag{3}$$
This cleanly splits responsibilities: the frame encoder learns *content*, the motion encoder learns *how content evolves*. It is explicitly framed as a learned latent-space analog of the motion vectors in classical video codecs (MPEG/H.264) — a keyframe plus inter-frame deltas.

**Preventing collapse (teacher–student).** The target for $\hat{z}_{t+1}$ is $z_{t+1}$ produced by the *same* frame encoder being trained, which invites trivial collapse to a constant. Following DINO, they keep two copies: a **student** updated by gradient descent and a **teacher** whose weights are an EMA of the student. The teacher produces the target $z_{t+1}^{\text{teacher}}$. Both pass representations through projection heads and a cross-entropy loss between prototype distributions penalizes collapse (identical distributions across frames raise the loss). Stop-gradients block the teacher from receiving gradients.

### Derivations / why it works
The load-bearing argument is conceptual rather than a formal proof. Two losses jointly enforce the causal principle while blocking degenerate solutions:

**Temporal prediction loss** ($\mathcal{L}_{\text{mse}}$) — supervises the causal constraint, that $z_t + \Delta z_t$ recovers the next embedding:
$$\mathcal{L}_{\text{mse}} = \lVert \hat{z}_{t+1} - \text{sg}(z_{t+1}^{\text{teacher}}) \rVert_2^2, \tag{4}$$
where $\text{sg}(\cdot)$ is stop-gradient, so only the motion encoder and student frame encoder are updated by this term, not the teacher.

**Self-distillation loss** ($\mathcal{L}_{\text{dino}}$) — prevents the collapse that $\mathcal{L}_{\text{mse}}$ alone would permit. A DINO-style cross-entropy between student ($p_s$) and teacher ($p_t$) projection distributions, applied over **both the [CLS] token and the patch tokens** (an extension over vanilla DINO, encouraging spatially consistent patch-level features):
$$\mathcal{L}_{\text{dino}} = -\sum_k p_t^{(k)} \log p_s^{(k)}, \tag{5}$$
where $k$ indexes the $K$ prototype dimensions of the projection head. The teacher distribution is centered with a running mean to prevent dimensional collapse. Temperatures are set $\tau_t = \tau_s = 0.1$.

**Total objective:**
$$\mathcal{L} = \lambda_{\text{mse}}\,\mathcal{L}_{\text{mse}} + \lambda_{\text{dino}}\,\mathcal{L}_{\text{dino}}, \tag{6}$$
with $\lambda_{\text{mse}} = \lambda_{\text{dino}} = 1.5$.

Why it avoids collapse where naive bias-removal fails: the MSE term gives a real predictive signal grounded in an exactly-correct constraint (next-frame prediction discards no information), and the EMA teacher's slow drift keeps student and teacher sufficiently different that trivial constant solutions are penalized by the cross-entropy term.

### Training procedure
- Pretraining data: **Something-SomethingV2 (SSV2)** — ~220k short clips of hand-object interactions, chosen for well-defined motion and manageable size.
- Architectures: ViT-S and ViT-B. TDV patch size 14 (DINO/iBOT baselines use 16).
- Optimizer AdamW, LR **1e-4** (baselines 5e-4), cosine schedule, warmup 0.5 epochs (baselines 10), weight decay 0.01, batch size 256 images.
- EMA momentum $\tau = 0.99$; student/teacher temperature 0.1/0.1; projection head dim 32768.
- ~200k steps (20 epochs); 2×NVIDIA H100 (80GB), ~48h per pretraining run.

### Inference / sampling
Not a generative model at deployment — the output is a frozen backbone used for downstream dense-prediction tasks. Notably, the design enables efficient video encoding: only the first frame needs the full frame encoder, with subsequent frames represented by composing the previous representation with the lightweight motion encoder (codec-style keyframe + deltas).

## Experimental setup

- **Motivation experiment:** models trained on ImageNet-1k subsets (0.1%, 1%, 10%, 100%), using masking ratio (10/30/50%) as a *continuous proxy* for assumption strength; metric = KNN accuracy.
- **Downstream tasks** (all pretrained on SSV2): semantic segmentation (UperNet on ADE20K, Cityscapes), optical flow (CroCo protocol; finetune on FlyingChairs/FlyingThings3D/MPI-Sintel, eval MPI-Sintel clean/final, EPE), stereo depth (SceneFlow-final, avg disparity error + bad-pixel rates @0.5px/1px).
- **Baselines:** DINO and iBOT (retuned on SSV2 for their max performance).
- Online ImageNet KNN Top-5 (k=20) used as a cheap collapse/quality proxy during training.

## Key results

- **Scale motivation (Figure 3):** at 0.1% data, 50% masking wins (30% trails by >12 pp); by 100% data, 30% masking *surpasses* 50%, and 10% masking approaches it. Optimal assumption strength decreases with scale.
- **Bias-removal collapse (Table 1):** stripping DINO's augmentations on SSV2 degrades KNN monotonically (24.63 → ... → 0.84 Top-1) and eventually collapses; full TDV reaches 8.79 Top-1 / 17.05 Top-5 *without* collapsing.
- **Optical flow (Table 3):** TDV beats both baselines on EPE, e.g. ViT-S MPI-Sintel clean 9.84 (TDV) vs 11.31 (iBOT) / 13.03 (DINO).
- **Stereo depth:** TDV has lower bad-pixel rates at 0.5px/1px across both archs (e.g. ViT-S bad@1px 39.70 vs 44.91/45.30), but slightly *higher* average disparity error (4.25 vs 3.50/3.64) — fewer large errors, but weaker in ambiguous regions.
- **Semantic segmentation (Table 2):** competitive but trailing by a small margin (ViT-S ADE20K mIoU 10.54 vs 10.60/10.71).

## Ablations

Two components are critical (their removal causes collapse): the **motion encoder** (KNN Top-5 drops to 1.87, collapses) and the **MSE loss** (drops to 1.58, collapses). Reducing TDV to just a DINO invariance loss over consecutive frames (no motion encoder) also fails — explicitly modeling temporal difference is necessary. Among non-collapsing choices: removing **centering** hurts most (17.05 → 11.15), then [CLS] in cross-attention (10.78), DINO loss on [CLS] (10.66); RoPE underperforms absolute positional encodings (10.25). Appendix documents failed alternatives: FiLM/AdaLN/AdaLN-Zero/Gated-AdaLN conditioning all showed early promise but collapsed; per-epoch teacher reinit hurt vs continuous EMA; continued-pretraining of MAE/DINOv2 with an unfrozen frame encoder degraded representations.

## Limitations

Authors' own: (1) TDV matches but does not exceed SOTA across the board; (2) weak on *semantic* benchmarks, attributed to the absence of crop-based invariance biases; (3) scaling beyond SSV2 did **not** help — Ego4D's erratic motion made the RGB-delta signal noisy, and FineVideo's scene cuts polluted the delta with editing transitions (pretraining on 2× FineVideo for the same 200k steps gave 10.75% vs 17.05% KNN, though training to 600k steps recovered to 16.04%), suggesting data quality/motion coherence matters. Honest-reader flag: results are at small scale (ViT-S/B, 2 H100s, one dataset), so the central "scales better" thesis is argued but not yet demonstrated for TDV itself.

## Why it matters [analyst's view]

This is a clean, principled entry in the "assume less" program, and its distinctive move is *modeling* the augmentation (temporal change) instead of enforcing invariance to it — which is exactly why it wins on dense/temporal tasks that need the discarded low-level information. The codec analogy is more than rhetoric: an encoder where only keyframes hit the full backbone and everything else is a cheap additive motion delta could be a genuinely efficient video representation substrate. The obvious next question is whether the modality-agnostic claim holds (audio, proprioception, touch) and whether it scales — the honest negative result on larger video data is the crux, and it's refreshingly reported. This connects to the broader JEPA lineage (LeCun is a coauthor) and to [[topics/self-predictive-learning]] — TDV is essentially a self-predictive objective with an explicit, non-discarded transition model.

## Open questions / things to verify

- Does the scale advantage actually materialize for TDV, or does the SSV2-tuned recipe not transfer? The data-quality sensitivity suggests the method may be more brittle than the "weak assumption" framing implies.
- Is causality really "exactly correct" as an *objective*? The claim is subtle (it's about not imposing invariance, not about determinism) — worth scrutinizing whether the EMA/centering machinery smuggles in its own biases.
- How does TDV compare to V-JEPA 2 and Midway Networks (its closest relatives) head-to-head at equal scale?

## Connections

- Topic MOCs: [[topics/representation-learning]], [[topics/self-supervised-learning]], [[topics/temporal-difference-learning]], [[topics/self-predictive-learning]], [[topics/jepa]]
- Author indices: [[authors/ninad-daithankar]], [[authors/alexi-gladstone]], [[authors/yann-lecun]], [[authors/heng-ji]]
- Related: closest prior work is Midway Networks (Hoang & Ren 2025) and MC-JEPA; builds on DINO (Caron 2021) and iBOT for the anti-collapse machinery; contrasts with pixel-reconstruction SSL (MAE). LeWorldModel ([[authors/lucas-maes]]) is cited as related temporal-prediction work.

## Selected quotes

> "What assumptions should our model have, if not reliant on existing inductive biases?" — §3.1

> "the 'augmentation' is induced by time, with temporally consecutive frames serving as the two views. The changes produced by this temporal augmentation are then modeled explicitly by the motion encoder, rather than being discarded via an invariance objective." — §3.1
