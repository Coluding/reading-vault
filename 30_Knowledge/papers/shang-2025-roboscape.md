---
type: paper
title: "RoboScape: Physics-informed Embodied World Model"
authors: ["Yu Shang", "Xin Zhang", "Yinzhou Tang", "Lei Jin", "Chen Gao", "Wei Wu", "Yong Li"]
year: 2025
venue: arXiv preprint
url: https://arxiv.org/abs/2506.23135
rw_id: 01ky5rktcnnhebph18qeb17x0d
topics: [world-models, robotics, video-generation]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# RoboScape: Physics-informed Embodied World Model

## TL;DR

RoboScape (Tsinghua + Manifold AI) is an action-conditioned, autoregressive embodied world model for robot manipulation that injects physical knowledge *implicitly* through two auxiliary joint-training tasks rather than through an external physics engine or cascaded simulator: (1) a temporal **depth-prediction branch** whose features are fused into the RGB branch to enforce 3D geometric consistency, and (2) **keypoint dynamics learning** — a self-supervised temporal-consistency loss on the visual tokens of the most active tracked keypoints, plus a keypoint-guided attention reweighting of the token loss — which implicitly encodes material properties (rigidity, deformability) without explicit material modeling. Trained on 50K clips from AgiBotWorld-Beta (~6.5M 16-frame training clips after slicing), it beats IRASim, iVideoGPT, Genie, and CogVideoX on all six metrics (e.g. LPIPS 0.1259 vs Genie's 0.1683; ΔPSNR action-controllability 3.34 vs 1.99). Downstream, Diffusion Policy trained purely on its synthetic data nearly matches real-data training on Robomimic Lift, and as a policy *evaluator* its success-rate judgments correlate with the ground-truth simulator at Pearson r = 0.953 where baselines are far lower.

## Context & motivation

Robot learning is bottlenecked by expensive real-world data collection (teleoperation), which world models promise to relieve by simulating environment dynamics — predicting future observations from past observations and actions — and thereby generating synthetic training data and scalable evaluation environments. But existing embodied world models (the paper names IRASim, iVideoGPT, Cosmos) optimize RGB pixels/tokens only. They produce visually plausible frames while violating physics: unrealistic object morphing, discontinuous motion, broken spatial consistency — worst in contact-rich manipulation with deformable objects (cloth), where small physical inconsistencies wreck downstream policies.

The paper sorts prior physics-injection attempts into three families and rejects each: **physics-prior regularization** (local rigidity / rotational-similarity constraints on Gaussian-splatting features or point clouds) is confined to narrow domains like human motion or rigid bodies; **simulator-based distillation** (physics engines producing motion signals or semantic maps as generation conditions) yields reliable priors but a computationally heavy cascaded pipeline; **material field modeling** is object-level and doesn't scale to scene-level generation. Recent joint RGB-depth world models (Aether, TesserAct) are the closest relatives, but the paper argues their learning stays at the whole-image level — missing fine-grained motion/deformation — and shows a trade-off where 3D perception gains cost RGB fidelity. RoboScape's claim: physics can be embedded *inside* the world model via cheap auxiliary supervision extracted by off-the-shelf annotators, no cascade needed.

## Method

### Problem formulation

Learn a dynamics function $f_\theta$ over robot-manipulation videos:

$$\mathbf{o}_{t+1} \sim f_{\theta}(\mathbf{o}_{t+1}\mid\mathbf{o}_{1:t}, \mathbf{a}_{1:t})$$

where $\mathbf{o} \in \mathbb{R}^{H\times W\times 3}$ is an RGB frame and $\mathbf{a} \in \mathbb{R}^{k}$ is a $k$-degree continuous action control vector (in experiments: concatenated end position, end orientation, and effector position). The model predicts the next frame given history and actions — an interactive, frame-level-controllable video world model.

### Core idea

Instead of bolting a physics engine onto a video generator, add two physics-informed *auxiliary training tasks* inside one autoregressive transformer: depth prediction supplies global 3D geometry, and keypoint-token temporal consistency supplies local motion/deformation (hence material) knowledge — both supervised by labels produced automatically by pretrained annotators, so physics comes essentially for free at training time and costs nothing extra at inference.

### Data pipeline (physical-priors annotation)

Built on AgiBotWorld; four stages, all with off-the-shelf models:

1. **Physical property annotation** — Video Depth Anything generates per-frame depth maps; SpatialTracker samples keypoints and tracks their trajectories. These two outputs are the physics supervision signals.
2. **Video slicing** — TransNetV2 detects camera/shot boundaries; Intern-VL assigns a single action semantic per clip, so each clip has consistent motion, no camera jumps, one action.
3. **Clip filtering** — FlowNet removes clips with indistinct or disordered motion; Intern-VL labels key frames and drops frames unrelated to them.
4. **Clip categorization** — clips organized by action difficulty and scene to support a curriculum-learning schedule (easy → hard tasks).

### Architecture: dual-branch co-autoregressive Transformer (DCT)

**Tokenization.** MAGVIT-2 compresses RGB frames $\mathbf{o}_{1:T} \in \mathbb{R}^{T\times H\times W\times 3}$ into discrete latent tokens $\mathbf{s}_{1:T} \in \mathbb{R}^{T\times H'\times W'\times D}$ with $H' = H/\alpha$, $W' = W/\alpha$ ($\alpha$ = spatial downsampling factor, $D$ = latent channel dim). Depth maps $\mathbf{d}_{1:T} \in \mathbb{R}^{T\times H\times W\times 1}$ are tokenized the same way into depth tokens $\mathbf{z}_{1:T}$.

**Backbone.** Two parallel branches $\mathcal{F}_{\text{RGB}}$ and $\mathcal{F}_{\text{Depth}}$, each a stack of Spatial-Temporal Transformer (ST-Transformer) blocks: **causal** attention in the temporal layers (so generation respects time order) and **bidirectional** attention in the spatial layers (full within-frame context). Conditioning is by learned action embeddings $\mathbf{c}_{1:t-1} = E_a(\mathbf{a}_{1:t-1}) \in \mathbb{R}^{(t-1)\times 1\times 1\times D}$ ($E_a$ = robot action encoder) plus position embeddings $\mathbf{e}_{1:t-1} \in \mathbb{R}^{(t-1)\times H'\times W'\times D}$, injected by simple element-wise addition with broadcasting ($\oplus$) — the paper reports additive fusion suffices for effective action control and keeps the model efficient:

$$\hat{\mathbf{s}}_t = \mathcal{F}_{\text{RGB}}(\mathbf{s}_{1:t-1} \oplus \mathbf{c}_{1:t-1} \oplus \mathbf{e}_{1:t-1}), \qquad \hat{\mathbf{z}}_t = \mathcal{F}_{\text{Depth}}(\mathbf{z}_{1:t-1} \oplus \mathbf{c}_{1:t-1} \oplus \mathbf{e}_{1:t-1}).$$

**Cross-branch geometry injection.** Depth knowledge only helps RGB rendering if it flows into the RGB branch. At every ST-Transformer block $l$, the depth branch's intermediate features $\mathbf{h}^{l}_{\text{depth}}$ are linearly projected and added to the RGB features:

$$\mathbf{h}^{l}_{\text{RGB}} \leftarrow \mathbf{h}^{l}_{\text{RGB}} + \mathcal{W}^{l}(\mathbf{h}^{l}_{\text{depth}})$$

where $\mathcal{W}^l$ is a learnable linear projection per block. This hierarchical fusion is *why* depth learning improves RGB geometry: the RGB branch continuously reads the depth branch's 3D-structure estimate at every layer rather than merely sharing a loss. Both branches are trained with token-level cross-entropy:

$$\mathcal{L}_{RGB} = -\sum_{t=1}^{T} \mathbf{s}_t \log p(\hat{\mathbf{s}}_t), \qquad \mathcal{L}_{Depth} = -\sum_{t=1}^{T} \mathbf{z}_t \log p(\hat{\mathbf{z}}_t)$$

($\mathbf{s}_t$, $\mathbf{z}_t$ ground-truth token indices; $p(\hat{\cdot}_t)$ the predicted token distributions).

### Keypoint dynamics learning (implicit material understanding)

Motivation: material properties (rigidity, elasticity) can't be learned by RGB pixel fitting alone, and physics engines are too costly/scene-specific. The insight is that *material understanding can emerge from self-supervised tracking of contact-driven keypoint dynamics* — e.g. correctly predicting how keypoints on a plastic bag move as an apple is placed inside implicitly captures the bag's material.

**Adaptive keypoint sampling.** SpatialTracker densely samples $N_0$ keypoints in the first frame and tracks coordinates across $T$ frames, yielding $\mathcal{T}_{dense} = \{(\mathbf{p}^1_i,\dots,\mathbf{p}^T_i)\}_{i=1}^{N_0}$, where $\mathbf{p}^t_i \in \mathbb{R}^2$ is keypoint $i$'s coordinate *in the tokenized feature map* of frame $t$. Instead of expensive segmentation masks to find contact regions, the model exploits the empirical observation that informative keypoints have large motion: it keeps the top-$K$ by motion magnitude $M_i = \sum_{t=1}^{T-1}\lVert \mathbf{p}^{t+1}_i - \mathbf{p}^{t}_i \rVert_2$, giving $\mathcal{T}_{sample}$ — typically points on the robot and manipulated objects.

**Keypoint token-consistency loss.** All frames are aligned to the initial frame: the predicted token at keypoint $i$'s location in frame $t$ should match the predicted token at its location in frame 1:

$$\mathcal{L}_{\text{Keypoint}} = \frac{1}{(T-1)K} \sum_{i=1}^{K} \sum_{t=2}^{T} \left\lVert \hat{\mathbf{s}}_{t}(\mathbf{p}_{i}^{t}) - \hat{\mathbf{s}}_{1}(\mathbf{p}_{i}^{1}) \right\rVert_{2}^{2}$$

where $\hat{\mathbf{s}}_t(\mathbf{p}^t_i) \in \mathbb{R}^D$ is the predicted token vector at keypoint $i$'s (tracked, hence moving) position in frame $t$. Because the ground-truth trajectory carries the point through the deformation, forcing token identity along the trajectory teaches the model *what stays the same as material moves and deforms* — an implicit material representation with no explicit material field.

**Keypoint-guided attention loss.** Dynamically active regions have higher token error (complex motion), so the token cross-entropy is reweighted there. A spatiotemporal map $\mathbf{A} \in \mathbb{R}^{T\times H'\times W'}$:

$$\mathbf{A}_{t,x,y} = \begin{cases} \gamma & \text{if } (t,x,y) \in \mathcal{T}_{sample} \\ 1 & \text{otherwise} \end{cases}, \qquad \mathcal{L}_{\text{Attention}} = -\sum_{t=1}^{T} \mathbf{A}_t \odot \mathbf{s}_t \log p(\hat{\mathbf{s}}_t)$$

with $\gamma$ a scalar importance weight ($\gamma = 5$ in experiments) and $\odot$ element-wise multiplication — i.e. hard-to-predict motion regions get $\gamma\times$ the loss weight.

### Joint objective

$$\mathcal{L} = \mathcal{L}_{RGB} + \lambda_1 \mathcal{L}_{Depth} + \lambda_2 \mathcal{L}_{Keypoint} + \lambda_3 \mathcal{L}_{Attention}$$

with tunable $\lambda_1, \lambda_2, \lambda_3 \in \mathbb{R}^{+}$; used values $\lambda_1 = 1$, $\lambda_2 = 0.01$, $\lambda_3 = 1$.

### Derivations / why it works

_No formal derivation; empirical paper._ The mechanism arguments are as above: depth features fused per-block act as a geometric prior on RGB generation; keypoint token consistency along tracked trajectories implicitly encodes deformation/material behaviour.

### Training procedure

- Data: 50,000 video clips from AgiBotWorld-Beta, 147 tasks, 72 skills; preprocessed into 16-frame clips sampled at 2 Hz → ~6.5M training clips.
- Actions: concatenated end position, end orientation, effector position.
- 5 epochs; $\lambda_1{=}1$, $\lambda_2{=}0.01$, $\lambda_3{=}1$, $\gamma{=}5$; ~24 h on 32× NVIDIA A800-SXM4-80GB. (Optimizer/lr/schedule _not addressed by the source_.)
- Model variants: RoboScape-S (34M), -M (131M), -L (544M).

### Inference / sampling

First frame given as conditional input; the model autoregressively predicts the subsequent 15 frames (long-horizon rollouts supported). Action commands condition each step, enabling interactive frame-level control. Only the RGB/depth transformer runs — no physics engine, no annotators at inference.

## Experimental setup

- **Datasets**: AgiBotWorld-Beta (training/eval of generation); Robomimic Lift and LIBERO task suite (downstream policy experiments).
- **Baselines**: embodied world models IRASim (DiT-based, action/trajectory-conditioned) and iVideoGPT (autoregressive interactive); general world models Genie (open-source 1XGPT reproduction) and CogVideoX (DiT text-to-video). Aether/TesserAct excluded — training code unavailable.
- **Metrics**: appearance fidelity — LPIPS (↓), PSNR (↑); geometric consistency — AbsRel depth error (↓), δ1/δ2 depth accuracy (↑); action controllability — ΔPSNR (↑, output sensitivity to the action condition).

## Key results

- **Generation (Table 1)**: RoboScape wins all six metrics — LPIPS 0.1259 / PSNR 21.85 (vs best baseline Genie 0.1683 / 19.76), AbsRel 0.360, δ1 0.6214, δ2 0.8307, ΔPSNR 3.34 (Genie 1.99; iVideoGPT 0.11; IRASim 0.03). CogVideoX generates pretty frames but can't follow actions; the embodied baselines degrade on long-horizon motion.
- **Policy training (Table 3)**: DP trained 10k steps on *only* generated data ≈ matches DP trained on real data on Robomimic Lift, with success rising monotonically as synthetic data grows (200→800 trajectories). π0 on LIBERO (multi-object, cluttered, long-horizon) with a 200-real-trajectory warm-up also improves steadily as generated data is added.
- **Policy evaluation (Fig. 5)**: rolling policies out inside the world model and scoring success (manual judgment) correlates with the ground-truth simulator at Pearson r = 0.953 for RoboScape; IRASim/iVideoGPT correlations are described as rather low.
- **Scaling (appendix)**: clear model-scaling law across S/M/L (all six metrics improve); data scaling (1M→6M clips) improves visual quality and controllability, but geometric metrics improve only marginally or slightly degrade — the paper attributes small-data geometric "wins" to overfitting to the final conditional frame.

## Ablations

Removing either physics task hurts, but the pattern is instructive (Table 2):

- **w/o depth**: AbsRel worsens 0.360→0.392, δ1 drops 0.621→0.579 — geometry degrades as expected; LPIPS/PSNR are actually marginally *better* (0.1249/21.95), confirming the RGB-vs-3D trade-off the paper criticizes in prior joint models exists here too, at small magnitude.
- **w/o keypoint**: ΔPSNR falls 3.34→2.95 and LPIPS worsens — keypoint learning carries action controllability and visual fidelity; notably its depth metrics (AbsRel 0.3417, δ1 0.6497, δ2 0.8673) *beat* the whole model, so the two tasks partially compete on geometry.
- **w/o both**: worst LPIPS (0.1299) and ΔPSNR collapses to 1.99 — the physics tasks together nearly double action controllability.
- Case study (Fig. 4): no depth → geometric distortions of moving objects; no keypoints → unreal motion patterns.

## Limitations

- **Paper's own**: future work is combining the model with real-world robots — i.e., all policy results are in simulation benchmarks (Robomimic, LIBERO), not physical deployment; synthetic-data diversity flagged as open in the Broader Impacts.
- **[analyst's view]** honest-reader flags: policy-evaluation success in the world model requires *manual judgment* of rollouts (subjective, unscaled); the two strongest recent competitors (Aether, TesserAct) were excluded for lack of code, so the joint-RGB-depth comparison is argued, not measured; ablations show the depth and keypoint tasks partially trade off on geometry vs fidelity, so "superior balance" depends on the λ weighting; physics is only ever evaluated via proxy metrics (depth accuracy, ΔPSNR, policy correlation), never by explicit physical-law checks; supervision quality is bounded by the off-the-shelf annotators (Video Depth Anything, SpatialTracker).

## Why it matters [analyst's view]

This is a clean exemplar of the *implicit* end of the physics-grounding spectrum the vault has been tracking: no physics engine, no material fields, no explicit physical constraints — just auxiliary prediction targets (depth, tracked keypoints) whose labels come from pretrained perception models, with cross-modal feature fusion doing the anchoring. [[papers/chen-2026-actionable-simulators]] cites exactly this as implicit physical anchoring via cross-modal consistency. It sits in contrast to the explicit-alignment approaches in the vault's physics-grounding cluster ([[papers/xiong-2026-physalign]], [[papers/yuan-2026-physics-alignment]]) and to diagnostic work on whether video models learn physics at all ([[papers/zhao-2026-phyworld]], [[papers/gu-2025-phyworldbench]]). The keypoint-token-consistency loss is the most transferable idea: a cheap, segmentation-free way to make a token-based generator respect object permanence and deformation — the paper's ΔPSNR ablation (1.99 → 3.34) suggests most of the action-controllability gain comes from these physics tasks, not the backbone. The policy-evaluator result (r = 0.953) is arguably the headline for robotics: a world model accurate enough to *rank policies* replaces expensive simulator/real evaluation loops.

## Open questions / things to verify

- How sensitive are results to $\lambda_2 = 0.01$ (keypoint loss is weighted 100× lower than the others) and to $K$, $N_0$, top-$K$ selection? Not ablated in the fetched text.
- Does keypoint token consistency hold up when objects rotate or change appearance drastically (token identity along a trajectory is a strong assumption)?
- Aether / TesserAct head-to-head, once code exists — is whole-image joint RGB-depth really worse?
- Real-robot transfer of policies trained on RoboScape data (the paper's own stated next step).
- Exact Table 3 per-column numbers (the HTML export scrambled the policy-learning table layout; re-check against the PDF before citing specific LIBERO percentages).

## Connections

- Cited as exemplar of implicit physical anchoring in: [[papers/chen-2026-actionable-simulators]]
- Physics-grounding cluster: [[papers/zhao-2026-phyworld]], [[papers/xiong-2026-physalign]], [[papers/yuan-2026-physics-alignment]], [[papers/xue-2026-acwm-phys]], [[papers/joseph-2026-physics-video-world-models]], [[papers/gu-2025-phyworldbench]]
- Vault synthesis: [[_synthesis/physics-grounding-video-world-models]]
- Action-controllable world models: [[papers/gao-2025-adaworld]]
- World models for policy training: [[papers/jiang-2025-world4rl]]
- Topic MOCs: [[topics/world-models]], [[topics/robotics]], [[topics/video-generation]]

## Selected quotes

> "we present RoboScape, a unified physics-informed world model that jointly learns RGB video generation and physics knowledge within an integrated framework" — Abstract

> "physical material understanding can emerge from self-supervised tracking of contact-driven keypoint dynamics" — §2.3

> "The results in Figure 5 show that the Pearson correlation of our model is 0.953, while the correlation of other models is rather low, indicating that our world model can be utilized as a better policy evaluator." — §3.4

> "While increasing data size consistently enhances visual quality and action controllability, geometric accuracy exhibits marginal improvement or even slight degradation." — Appendix C
