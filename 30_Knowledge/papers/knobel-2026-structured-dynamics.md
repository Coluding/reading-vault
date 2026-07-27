---
type: paper
title: "Self-Supervised Learning of Structured Dynamics from Videos"
authors: ["Lukas Knobel", "Andrew Zisserman", "Yuki M. Asano"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2607.21576
rw_id: 01kyhvgzqa5bdsh0b28yqa5jzj
topics: [self-supervised-learning, representation-learning, world-models]
priority: high
read_state: queued
relevance: ""
added: 2026-07-27
last_updated: 2026-07-27
---

# Self-Supervised Learning of Structured Dynamics from Videos (SDM)

## TL;DR

Frame-to-frame change in video entangles two causes — **camera motion** and **object motion** — and the latent-action / world-model literature almost always squashes both into a *single* transition token. SDM (Structured Dynamics Model) asks whether a **frozen** image backbone already contains enough signal to separate them, and answers yes: it stacks a two-stage recurrent module on frozen DINOv2 features that predicts the *next feature map*, factoring the transition into a **primary** token $p_t$ (the dominant source of change) and a **residual** token $r_t$ (whatever's left). The separation is induced not by architecture alone but by *selectively applying the loss* under two cheap scene-level labels available for free in synthetic Kubric data — "static scene" and "static camera" — while real video trains unlabelled. On **ProbeMotion**, a new 7-task linear-probe suite, SDM beats CLS/AVG-pool frozen-backbone descriptors nearly everywhere, beats [[papers/kerssies-2026-delta-tokens]] (DeltaTok) on 5/7 tasks *despite 8× fewer training iterations*, and surpasses strongly-supervised 3D models (VGGT-1B, Depth Anything 3, Pi3X) on 3/7 — at 215M parameters (87M of it frozen) versus their ~1B, and with no camera-pose, depth, or point-cloud supervision. Total cost: ~11h on 4×H100.

## Context & motivation

Self-supervised image backbones (DINOv2, MAE, SigLIP) carry rich semantic and geometric structure, and modern 3D systems build on them — but they still lean on heavy supervision during adaptation: 3D annotations, structure-from-motion pseudo-ground-truth, geometric labels. The paper's pointed example: **VGGT trains on 17 annotated datasets.** That limits scalability and leaves the actual question unanswered — *how much scene dynamics can be recovered from generic pretrained visual features without dense supervision?*

Video is the obvious free supervision, but it has a structural problem: frame-to-frame change mixes camera-induced and object-induced dynamics, so a standard self-supervised prediction objective learns *aggregate* change without separating causes. The paper's specific complaint about the latent-action line (which the vault holds via DeltaTok and AdaWorld) is that "many latent action or world models summarize scene change with a single latent token, mixing dominant global change with residual object-centric dynamics." A single transition token has to explain everything at once — appearance-dependent feature variation *and* low-dimensional motion factors like camera translation or object direction.

SDM's bet is the opposite of end-to-end video pretraining: keep the image backbone frozen, and impose the decomposition explicitly on top. The motivating intuition is stated plainly — "much of the temporal change in videos is governed by lower-dimensional dynamics acting on high-dimensional visual features."

## Method

### Problem formulation

Given a frozen image encoder $E$ producing per-frame patch feature maps $f_t$, learn a module that predicts $f_t$ from $f_{t-1}$ while routing the transition through two low-dimensional latent tokens $p_t, r_t \in \mathbb{R}^D$ that respectively capture the dominant and the residual source of change. The learned representation used for all downstream evaluation is the pair $(p_t, r_t)$ — never the predicted features themselves.

### Core idea

Predict the future feature map through **two sequential compensation stages** rather than one. The first stage is given the *opportunity* to explain the whole transition; the second only ever sees what the first failed to explain. Chain that structural asymmetry to a loss that is *applied differently depending on which kind of motion the clip is known to contain*, and the two tokens specialise without either ever being told what "camera motion" is.

### Architecture / algorithm

Two stages, each a (motion extractor $\phi$, feature predictor $\psi$) pair.

**Primary motion extraction.** A recurrent token $p_t \in \mathbb{R}^D$ is updated by attending to the current frame pair:

$$p_t = \phi_p(p_{t-1};\, f_{t-1}, f_t) \tag{2}$$

with $p_0$ a learnable parameter. The recurrence is what lets the model accumulate motion information across a clip while each individual update only ever compares *adjacent* frames — so the token integrates over time without the extractor needing a long context window. The paper notes these tokens are "autoregressive but non-causal across time."

**Primary motion compensation.** The predictor uses $p_t$ to transform the *source* feature map:

$$f_t^{>} = \psi_p(f_{t-1};\, p_t) \tag{3}$$

$f_t^{>}$ ("f modified once") is an intermediate, primary-motion-compensated representation. Note what is *not* happening: $\psi_p$ never sees $f_t$. It must push $f_{t-1}$ toward the target using only the low-dimensional $p_t$ as instructions, which is exactly the bottleneck that forces $p_t$ to actually carry the motion rather than letting the predictor copy the answer.

**Residual motion extraction.** Whatever gap remains between $f_t^{>}$ and $f_t$ is by construction the part the primary component could not explain:

$$r_t = \phi_r(r_{t-1};\, f_t^{>}, f_t) \tag{4}$$

with $r_0$ learnable. Conditioning on $f_t^{>}$ rather than $f_{t-1}$ is the load-bearing choice — it is what makes $r_t$ *residual* rather than a second, redundant copy of the full transition.

**Residual motion compensation.** The final prediction refines the intermediate:

$$f_t^{\gg} = \psi_r(f_t^{>},\, r_t) \tag{5}$$

The pair $(p_t, r_t)$ is the structured motion representation. Extension beyond frame pairs is trivial: apply the same updates each timestep, carrying $p_{t-1}, r_{t-1}$ forward, so temporal context accumulates in the tokens.

### Training objective — where the structure actually comes from

This is the paper's real mechanism, and it is less about architecture than it looks. All losses are MSE in the frozen encoder's feature space, $\mathcal{L}_{\text{MSE}}(x,y) = \frac{1}{HWD}\|x-y\|_2^2$ (normalised by height, width, and channel count so the scale is comparable across feature-map sizes). What differs is **which intermediate gets supervised**, keyed on two weak scene-level annotations that Kubric provides for free:

1. **Unlabelled or fully dynamic clips** — supervise only the end of the chain:
   $$\mathcal{L} = \mathcal{L}_{\text{MSE}}(f_t^{\gg},\, f_t)$$
   No structural constraint is imposed; the model just has to predict well. This covers all real video.

2. **Static scene** (camera moves, nothing else does) — the primary component *should* explain the entire transition, so bypass the residual stage and supervise the intermediate directly:
   $$\mathcal{L} = \mathcal{L}_{\text{MSE}}(f_t^{>},\, f_t)$$
   This is the positive anchor: it teaches $p$ what pure camera-induced change looks like, because in these clips there is nothing else for it to be.

3. **Static camera** (scene moves, camera doesn't) — there is no global change to compensate, so the primary stage is *regularised to do nothing* while the residual stage picks up the scene dynamics:
   $$\mathcal{L} = \mathcal{L}_{\text{MSE}}(f_t^{\gg},\, f_t) + \lambda_{\text{reg}}\,\mathcal{L}_{\text{MSE}}(f_t^{>},\, f_{t-1})$$
   The second term is the negative anchor, and it is the clever one: it penalises $\psi_p$ for changing anything, pushing $f_t^{>} \to f_{t-1}$ (identity), so object motion is forced downstream into $r$.

Together these are two-sided pressure — one label says "primary explains everything," the other says "primary explains nothing" — and the unlabelled majority just has to predict. Neither token is ever given a motion *label*; they are only told, on a quarter of the data, which stage is allowed to do the work.

### Implementation details

- **Backbone**: frozen DINOv2-B/14 with registers. Patch features from **all 12 blocks** are concatenated channel-wise and projected by a 2-layer MLP down to $D=768$ (one block's hidden width) before SDM; predictions are linearly projected back to the concatenated multi-layer space before the loss. Using intermediate features, not just the final layer, is ablated as a real win.
- **Motion extractors $\phi_p, \phi_r$**: 4-block transformer decoders with $n_{\text{reg}}=4$ learnable registers. The motion token and registers self-attend and cross-attend to frame features, with a 3D RoPE extension (spatial + temporal positions, base 100). Initial tokens drawn from $\mathcal{N}(0, 0.02^2)$.
- **Predictors $\psi_p, \psi_r$**: shallower 2-block decoders that cross-attend to their motion token, 2D RoPE. Deliberately weaker than the extractors — the predictor is a decoder of the token, not a second encoder of the scene.
- **Training**: AdamW, 200k iterations, batch size 128, lr $6.25\times10^{-5}$, 1k warmup, weight decay 0.05, $\lambda_{\text{reg}} = 0.5$. 5 frames at 2 fps, $224\times224$, **no data augmentation**. ~11h on 4×H100.
- **Data mixture**: Kubric / SSv2 / DL3DV at fixed ratios 0.5 / 0.25 / 0.25. 180k Kubric sequences (half static-scene, the dynamic half split evenly between moving- and static-camera), ~170k SSv2 clips and ~4k DL3DV clips used **without labels**. Net effect on each minibatch: **1/4 gets static-scene supervision, 1/8 gets static-camera supervision, 5/8 is plain final-prediction only.** The structural signal is a minority of the data.
- **Size**: 215M parameters total including the 87M frozen backbone.

## Experimental setup

- **ProbeMotion** (introduced here): 7 linear-probe tasks over Kubric (synthetic, disjoint train/test objects and backgrounds), DL3DV and CameraBench (real camera motion), static-camera subsets of DAVIS2017 and YouTubeVOS (2D object motion), and SSv2-110k (motion-heavy action semantics). Static subsets are built by filtering clips on VGGT-estimated camera motion and labelling object motion from mask-centroid displacement.
- **Metrics**: normalised-MSE regression on 6-DoF camera motion (Kubric, DL3DV), 3D translation for Kubric object motion (simulated objects don't rotate), 2D displacement regression for DAVIS/YouTubeVOS, top-1 accuracy for CameraBench and SSv2-110k. All probes are **single linear layers on frozen features**, 3 seeds.
- **Baselines**: frozen DINOv2 CLS (concatenated) and AVG-pool (feature differences), both also given multi-frame and all-12-block variants; **DeltaTok** (self-supervised single delta token, trained 8× longer at higher resolution — the paper calls it a strong baseline); and the supervised-geometry tier **VGGT-1B, Depth Anything 3, Pi3X** (~1B params each, trained with camera pose / depth / point-cloud supervision).
- Probing convention: $p$ for camera motion (Kubric, DL3DV, CameraBench), $p$ for object motion where the camera is static (DAVIS/YouTubeVOS) and for SSv2 action recognition (a VGGT analysis shows SSv2 has low inter-frame camera motion), $r$ for Kubric object motion under a moving camera.

## Key results

- **vs frozen-backbone descriptors**: SDM wins on every ProbeMotion task except static DAVIS2017 (where AVG-pool is best on 2D displacement — attributed to that set being small, curated, and image-plane-simple). Gaps are largest on the 3D probes (Kubric, DL3DV), and hold even when baselines get all 12 DINOv2 layers — so the gain is from structured future-feature prediction, not from richer frozen features.
- **vs DeltaTok**: better on **5 of 7**, largest margins on Kubric object motion (**0.16 lower MSE**) and SSv2-110k (**+9.6 p.p.**). DeltaTok wins static DAVIS2017 and CameraBench. Notable given DeltaTok's 8× larger iteration budget and higher input resolution.
- **vs supervised 3D models**: best on **2 of 3 object-motion probes**, beating the strongest supervised baseline by **0.13 MSE** on Kubric object motion and **0.04 MSE** on static YouTubeVOS; approaches them on DL3DV, comparable on CameraBench, behind VGGT/Pi3X on static DAVIS2017. Overall best on 3/7 despite ~5× fewer parameters and no geometric supervision.
- **SSv2-110k** (the generalisation probe): SDM beats CLS by **+8.2 p.p.**, AVG-pool by **+20.8 p.p.**, DeltaTok by **+9.6 p.p.**, and the best supervised descriptor (Pi3X average-feature concatenation) by **+3.3 p.p.** — while the Pi3X probe has twice SDM's input dimension.

## Ablations

- **Token swapping (Table 5) — the specialisation check.** On dynamic-camera data $p$ is substantially better than $r$ for camera motion, while $r$ is best for Kubric object motion. Under static cameras, where object motion *becomes* dominant, $p$ takes over on static YouTubeVOS and on SSv2 action recognition. This is the paper's strongest claim and it is well-designed: the roles are not "camera token" and "object token" but **dominant** and **residual**, and the assignment tracks whichever source dominates that clip. Concatenating $[p,r]$ barely helps, confirming task information concentrates in one token — the exception being Kubric object motion in world coordinates, which legitimately needs both. Static DAVIS2017 is flagged as unstable across seeds for $r$ ($0.8 \pm 0.15$).
- **Temporal context (Table 3)**: Kubric camera MSE $0.26 \to 0.16$ and object MSE $0.22 \to 0.19$ going from $T=2$ to $T=4$; CameraBench $84.7\% \to 85.3\%$; SSv2-110k rises monotonically $16.7\%$ at $T=2$ to **$23.1\%$ at $T=7$ — beyond the $T=5$ training range**. Crucially, the multi-frame AVG-pool baseline *at maximum context* still underperforms SDM at a **single frame pair**, so the gains aren't just "more frames."
- **Both weak anchors are needed**: removing either scene-level constraint weakens the primary/residual structure; making extraction joint rather than sequential makes the tokens encode overlapping motion.
- **$\lambda_{\text{reg}}$ trades off** Kubric object-motion probing against SSv2 action recognition; 0.5 is a balanced default (i.e. this is a tuned compromise, not a free win).
- **Backbones (Table 4)**: works across MAE, SigLIP, DINOv2r, DINOv3; DINOv2/3 best.
- **Intermediate features and real data both help** — multi-layer beats final-layer, and unlabelled real video improves generalisation beyond synthetic-only training.

**Qualitative**: stage-wise error maps show the primary stage correcting global camera-induced change while *increasing* error around an independently moving car, which the residual stage then fixes — a clean visual confirmation of the intended division. Repeatedly applying the last motion token gives plausible short-horizon extrapolation that drifts over longer horizons; motion tokens transferred from a source clip to a target clip with a different object induce consistent short-horizon motion (also drifting).

## Limitations

**Paper's own:** the decomposition is anchored by weak scene-level supervision on *synthetic* data — replacing it with fully self-supervised constraints would improve scalability. ProbeMotion's static subsets rely on VGGT-estimated camera motion for filtering, which introduces label noise. And linear probes measure only *linearly decodable* information.

**An honest reader would add:**
- The evaluation is entirely linear probing on frozen tokens. No downstream control, planning, or generation task is attempted, so "useful inductive bias for learning and analyzing latent video dynamics" is asserted more than demonstrated — the world-model framing writes a cheque the experiments don't cash.
- The Kubric-derived anchors mean the primary/residual semantics are grounded in synthetic camera/scene statistics; whether the same division survives on real footage with handheld jitter, rolling shutter, or multiple independently-moving objects is untested. The "dominant source" abstraction plausibly degrades when there is no single dominant source.
- Motion extrapolation and token swapping both "drift at longer horizons," which is stated but never quantified.
- SDM is compared against supervised 3D models on *motion probes chosen by this paper*; VGGT and friends are built for reconstruction, so the comparison flatters SDM on axes those models weren't optimised for. The paper is reasonably upfront that they're included "to contextualize performance under much stronger supervision."

## Why it matters [analyst's view]

The genuinely interesting contribution is not the architecture but **the supervision design**: getting a semantic factorisation out of a model using two binary scene-level flags on a quarter of the batch, with no motion labels anywhere. The trick — supervise a *different intermediate* depending on what the clip is known to contain, including a term that explicitly penalises the first stage for doing anything — is a reusable pattern for inducing interpretable structure in latent variables. It sidesteps the usual disentanglement machinery (adversaries, VAE-style factorised priors, contrastive negatives) entirely, and the token-swap table is the kind of evidence disentanglement papers usually can't produce.

For the vault this lands directly against [[papers/kerssies-2026-delta-tokens]]: DeltaTok compresses a frame to *one* delta token for efficient generative world modelling, and SDM is the argument that one token is the wrong number when the change has two causes. Beating it on 5/7 probes with 8× less compute is a real result, though the objectives differ enough (generation vs probing) that it isn't a clean refutation. It also sits alongside [[papers/gao-2025-adaworld]] in the latent-action line, and next to [[papers/baldassarre-2025-dino-world-models]] as another "freeze DINOv2, train only the dynamics module" recipe — a pairing worth noting, since DINO-world freezes the encoder to *avoid collapse* while SDM freezes it to *test what's already there*. Same architectural move, different scientific question.

The frozen-backbone result is the broader claim: pretrained image models already encode enough for structured motion, so a great deal of video-representation work may be re-derivable as lightweight modules on frozen features rather than end-to-end video pretraining. That's the same efficiency thesis running through [[papers/zhu-2026-sana-wm]], arrived at from the representation side rather than the architecture side.

## Open questions / things to verify

- Does the primary/residual split survive multi-object scenes with no dominant motion source? Kubric's controlled setup may be doing more work than acknowledged.
- Can the weak anchors be replaced by a self-supervised proxy — e.g. deriving "static camera" from optical-flow global-motion statistics rather than simulator ground truth? That's the paper's own stated next step and the main scalability blocker.
- Would a third stage (or $k$ stages) keep factorising usefully, or does the residual become noise immediately?
- The extrapolation and token-swapping demos suggest $(p, r)$ could act as a *control* interface for a latent world model. Untested — and it's the experiment that would substantiate the world-model framing.
- How much of the SSv2 gain is motion structure versus simply having a learned temporal aggregator? The multi-frame baseline comparison partly addresses this but not fully.
- **VGGT** [35], **Depth Anything 3** [20], **Pi3X** [39], and **CroCo/SiamMAE** [15, 40] are all load-bearing comparisons — `_needs note_`.

## Connections

- Builds on: DINOv2 with registers (frozen encoder), Mur-Labadia et al. (multi-layer feature concatenation) — `_needs note_`.
- Topic MOCs: [[topics/self-supervised-learning]], [[topics/representation-learning]], [[topics/world-models]]
- Related in vault:
  - [[papers/kerssies-2026-delta-tokens]] — **the direct baseline and the direct foil**: one delta token per frame vs SDM's structured pair. Read these two together.
  - [[papers/baldassarre-2025-dino-world-models]] — same frozen-DINOv2 + learned-predictor recipe, different motivation (collapse avoidance vs probing what's already encoded).
  - [[papers/gao-2025-adaworld]] — latent-action world modelling, the line SDM criticises for collapsing change into a single latent.
  - [[papers/daithankar-2026-temporal-difference-vision]] — also learns image representations from video temporal difference; complementary angle on the same signal.
  - [[papers/ivashkov-2026-sensorimotor-world-models]] — another "what does the latent actually capture?" paper, with inverse dynamics rather than weak anchors as the structuring mechanism.
- Author index: [[authors/lukas-knobel]]

## Selected quotes

> "Frame-to-frame change entangles the two sources of dynamics, camera motion and object motion. Standard self-supervised prediction objectives can therefore model aggregate change without separating its underlying causes." — §1

> "A single transition token must explain all sources of change, from appearance-dependent feature variation to low-dimensional motion factors such as camera translation or object direction." — §2.2

> "The primary stage corrects the dominant global change, including the camera-induced movement of the edges of the track. Notably, primary motion compensation increases the error around the independently moving car. The residual stage reduces this localized error." — Figure 3 caption
