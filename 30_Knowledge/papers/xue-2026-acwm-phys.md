---
type: paper
title: "ACWM-Phys: Investigating Generalized Physical Interaction in Action-Conditioned Video World Models"
authors: ["Haotian Xue", "Yipu Chen", "Liqian Ma", "Zelin Zhao", "Lama Moukheiber", "Yuchen Zhu", "Yongxin Chen"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2605.08567
rw_id: ""
topics: [world-models, video-generation, robotics]
priority: medium
read_state: skimmed
relevance: ""
added: 2026-07-10
last_updated: 2026-07-10
---
<!-- ingested via /paper-search from arXiv (not via Readwise; no rw_id) -->

## TL;DR

ACWM-Phys is a benchmark for **action-conditioned world models (ACWMs)** that probes whether diffusion/flow-matching video predictors actually learn physics or just appearance statistics. It spans **eight robotic simulation environments** grouped into **four physical interaction categories** — rigid-body, deformable-object, particle, and kinematics — with **15k+ trajectories** (paired images, actions, labels) and **controlled in-distribution (InD) vs out-of-distribution (OoD)** splits where the shift is along a physically meaningful axis (e.g. unseen cube counts, longer ropes, more particles, expanded workspaces). The authors train a strong DiT-based baseline, **ACWM-DiT** (frozen WanVAE latents + flow matching), across all environments. The headline finding: **OoD generalization is driven primarily by task complexity, not physics category** — low-DoF geometric tasks (Push Cube, Reacher) stay nearly stable OoD while high-DoF kinematics (Robot Arm, ΔM-MSE +40.35) and contact-rich deformation (Cloth Move, ΔM-MSE +29.99) degrade sharply, evidence that "models still rely heavily on appearance statistics rather than internalizing physics." Design ablations show cross-attention conditioning helps only for high-dimensional actions, and temporally-aware (causal) VAEs beat frame-wise encoders.

## Context & motivation

Action-conditioned world models predict future frames given past observations and a planned action sequence, and have shown promise for video prediction and decision-making. But the paper argues existing ACWMs and their benchmarks "suffer from a critical blind spot: *physical diversity*." Per the abstract, prior benchmarks "are largely restricted to egocentric navigation or narrow, task-specific robotics datasets, offering only limited coverage of the rich physical interactions required for generalized world understanding." Most current work is "confined to egocentric navigation or to narrow robot manipulation involving mostly rigid-body pick-and-place."

The contribution is a benchmark that deliberately spans **four distinct physical interaction regimes** with **controlled InD/OoD evaluation**, a strong DiT baseline (ACWM-DiT), the empirical finding that generalization tracks task complexity rather than physics category, and design insights on conditioning, latent compression, and action-space dimensionality. Environments are built in a fully controllable simulator to enable "precise data collection, reproducible evaluation, and systematic analysis of model capabilities for physically grounded world modeling," using the **PyFlex** and **MuJoCo** physics engines.

## Method

### Problem formulation

Given past observations $\mathbf{o}_{1:t}$ and future actions $\mathbf{a}_{t:t+h-1}$, the model learns the conditional distribution over future frames:

$$p(\mathbf{o}_{t+1:t+h} \mid \mathbf{o}_{1:t}, \mathbf{a}_{t:t+h-1})$$

or, in latent space, $p(\mathbf{z}_{t+1:t+h} \mid \mathbf{z}_{1:t}, \mathbf{a}_{t:t+h-1})$ where $\mathbf{z}_t = \mathcal{E}(\mathbf{o}_t)$ is the VAE encoding of frame $\mathbf{o}_t$. Here $t$ is the number of context frames, $h$ the prediction horizon. "Generalized physical interaction" means the model must produce faithful rollouts across qualitatively different dynamics (rigid, deformable, granular, articulated) and under controlled distribution shifts along physical axes, not just within the training regime.

### Core idea

Rather than proposing a new model, the paper builds a **controlled benchmark** that disentangles *physics category* from *task complexity*, then trains a common baseline (ACWM-DiT) across all environments so that generalization behavior can be attributed to properties of the interaction rather than to modeling idiosyncrasies. The InD/OoD splits are engineered so each OoD shift moves along one physically meaningful axis.

### Benchmark construction

Eight environments across four categories (each action's dimensionality noted):

- **Rigid-body dynamics** — *Push Cube*: moves one to five colored cubes with a circular pusher (2D position action); *Stack Cube*: a Franka Panda places a red cube on a green cube (7-DoF end-effector pose + gripper).
- **Deformable-object dynamics** — *Push Rope*: a pole pusher deforms a flexible rope in PyFlex (2D displacement); *Cloth Move*: pushes a cloth over a fixed sphere using dual arms (3D displacement).
- **Particle dynamics** — *Push Sand*: rearranges granular material in PyFlex with a board pusher (7-D board pose); *Pour Water*: pours fluid by moving and tilting a cup (4-D Cartesian + tilt).
- **Kinematics** — *Robot Arm*: 7-DoF Franka Panda controlled by per-joint angle deltas; *Reacher*: a two-link MuJoCo arm controlled by joint torques.

Dataset: "more than 15k simulated trajectories with paired image observations, actions, and evaluation labels," with both InD and OoD splits.

**OoD distribution shifts (one physical axis per env):**
- Rigid: Push Cube tests unseen cube counts; Stack Cube shifts target placement.
- Deformable: Push Rope changes rope length; Cloth Move varies cloth size.
- Particle: Push Sand increases particle count; Pour Water shifts water level.
- Kinematics: Robot Arm expands the goal workspace; Reacher tests unseen goal regions.

### Evaluation protocol / metrics

Each environment has separate InD train/test splits plus an OoD test split with a controlled shift. Metrics: **MSE, SSIM, PSNR**, plus a motion-aware **Masked-MSE (M-MSE)** that "computes MSE only on pixels with sufficient ground-truth temporal change, emphasizing motion-relevant regions while down-weighting static backgrounds":

$$\text{M-MSE} = \frac{\sum_{t,c,h,w} w_{h,w}\,(\hat{o}_{t,c,h,w} - o_{t,c,h,w})^2}{\sum_{t,c,h,w} w_{h,w}}$$

with per-pixel weight $w_{h,w} = 0.01 + m_{h,w}$ and motion map $m_{h,w} = \max_{t\in[1,T],\,c} |o_{t,c,h,w} - o_{1,c,h,w}|$. Symbols: $\hat{o}$ predicted pixel, $o$ ground-truth pixel, indices $t$ (time), $c$ (channel), $(h,w)$ (spatial); $m_{h,w}$ is the max absolute deviation of any pixel from the first frame, so static background gets weight ≈0.01 and moving regions get up-weighted.

### Model / architecture studied

**ACWM-DiT baseline:**
- **Latent encoder**: frozen **WanVAE** (8× spatial, 4× temporal compression, 16 channels). Fixed latent sequence length $T_l = 37$ tokens.
- **Backbone**: DiT with alternating spatial/temporal self-attention and RoPE positional encoding. Two scales: **DiT-S** (~200M params, 768 hidden dim, 10 layers) and **DiT-M** (~600M params).
- **Action conditioning**: each action vector is projected from its env-specific dimension $d_a$ to the DiT hidden dim $d$ by an MLP (Linear→SiLU→Linear), then a **1D strided temporal convolution** (kernel size 3, stride $r=4$, matching the VAE's 4× temporal compression) downsamples the pixel-rate action sequence to latent temporal resolution.
- **Objective**: flow matching with 1000 noise levels, shift parameter $s = 5.0$:

$$\mathcal{L}_{\text{ACWM}} = \mathbb{E}\,\big\|\,\mathbf{v}_\theta(\mathbf{z}_\tau, \tau, \mathbf{h}_t) - \dot{\alpha}(\tau)\mathbf{z}_0 - \dot{\beta}(\tau)\mathbf{z}_1\,\big\|_2^2$$

where $\mathbf{v}_\theta$ is the predicted velocity field, $\tau$ the flow time, $\mathbf{z}_\tau$ the interpolated latent, $\mathbf{h}_t$ the conditioning (context latents + embedded actions), $\mathbf{z}_0/\mathbf{z}_1$ the endpoints of the interpolation path, and $\dot\alpha,\dot\beta$ the time-derivatives of the interpolation schedule coefficients.

### Training procedure

AdamW, learning rate 1e-4, gradient clipping at 1.0; 100k steps; batch size 4 on 8× H100 GPUs **per task**. Input resolution 240×240 (240×400 for Push Sand). Inference: 50 denoising steps.

### Inference / sampling

Generation is flow-matching denoising with 50 steps; the paper notes inference is **bidirectional** (not causal/streaming), which is flagged as a limitation for real-time use.

## Experimental setup

- **Datasets**: the eight ACWM-Phys environments (15k+ trajectories), each with InD and OoD splits.
- **Baselines / comparisons**: the paper's own ACWM-DiT at two scales (DiT-S, DiT-M) plus ablation variants (AdaLN vs cross-attention conditioning; WanVAE vs FLUX VAE latents; action-dimensionality variants; data-fraction variants).
- **Metrics**: MSE, PSNR, SSIM, M-MSE; InD vs OoD, and the OoD gap (ΔM-MSE, ΔSSIM).

## Key results

*(Summarized — see paper tables for full grids.)*

- **Strong InD fidelity for simple/repetitive dynamics**: Push Rope M-MSE 2.61 / SSIM 0.988; Reacher M-MSE 5.63 / SSIM 0.992.
- **Hard InD cases**: Cloth Move M-MSE 63.68 / SSIM 0.920 ("large-scale deformation leads to much larger errors"); Robot Arm M-MSE 13.45.
- **OoD degradation tracks complexity, not physics category**: Robot Arm shows the largest drop (ΔM-MSE +40.35, ΔSSIM −0.067); Cloth Move ΔM-MSE +29.99, ΔSSIM −0.056; whereas **Push Cube and Reacher remain nearly stable**. The takeaway: "OoD robustness is shaped by both physical complexity and action/state dimensionality," and models "capture visual statistics rather than physical laws" / "still rely heavily on appearance statistics rather than internalizing physics."

## Ablations

This is the load-bearing section — the design insights:

- **Model scaling (Table 2)**: DiT-S → DiT-M on Cloth Move improves MSE 10.776 → 10.248 (InD) and 22.416 → 20.031 (OoD); scaling "consistently improves both InD and OoD performance, with larger gains on OoD."
- **Action conditioning: cross-attention vs AdaLN (Table 3)**: for **low-dimensional** actions ($d_a = 2$) AdaLN is comparable to cross-attention; for **high-dimensional** Robot Arm ($d_a = 7$) cross-attention substantially wins — MSE 1.434 → 0.691 (InD), 6.559 → 4.596 (OoD). "Cross-attention is most useful when actions are high-dimensional and require structured spatial-temporal grounding."
- **Latent formulation: temporal VAE vs frame-wise (Table 4)**: **WanVAE** (4× temporal compression) beats **FLUX VAE** (1× temporal, i.e. frame-wise) on Pour Water (InD MSE 2.630 vs 3.241) and Robot Arm (1.434 vs 2.314). "Temporally-aware latent representations are beneficial" — i.e. causal/temporal encoders > frame-wise encoders.
- **Action-space dimensionality (Table 7)**: richer action signals help where dynamics are genuinely high-DoF but hurt where they aren't. Cloth Move 8-DoF beats 3-DoF on OoD (MSE 6.922 vs 23.820) — "richer action signals help the model infer two-arm cloth dynamics." Conversely, Push Cube with two pushers is *worse* (InD MSE 6.034 vs 2.919 single).
- **Data efficiency (Table 5)**: Pour Water degrades gracefully at 50% data (−0.26 dB) while Push Cube degrades sharply — "data efficiency is governed by geometric variability."

## Limitations

Paper's own: ACWM-DiT is **not real-time** because of bidirectional diffusion inference; the benchmark is **simulation-based** and "does not fully capture the complexity of real-world physics, sensing, and robot interaction." [analyst's view] Additional honest flags: all results are on a single family of DiT baselines (no comparison to external published ACWMs), so "current models rely on appearance" is inferred from this baseline's behavior rather than a broad model sweep; and "physics category vs complexity" are partly confounded since higher-DoF tasks also tend to have richer visuals.

## Why it matters [analyst's view]

This paper reframes the "do video world models know physics?" question with a **controlled experimental scalpel**: by fixing the model and varying physics category vs task complexity independently, it produces a cleaner attribution than pure faithfulness benchmarks. The result — generalization tracks DoF/complexity, not physics regime, and models lean on appearance statistics — is a concrete, testable diagnosis rather than a vibe. For robotics, it directly bears on whether an ACWM can serve as a planner/simulator: it can within-distribution for low-DoF geometric tasks, but degrades exactly where manipulation gets interesting (deformables, articulated high-DoF arms). The conditioning ablation (cross-attention only pays off for high-$d_a$ actions) and the temporal-VAE result are immediately actionable architecture guidance for anyone building an ACWM. It sits squarely in the physics-faithfulness cluster ([[papers/zhao-2026-phyworld]], [[papers/lin-2026-phyground]]) but adds the *action-conditioned* axis those lack, and it provides an empirical counterweight to policy-oriented world-model work ([[papers/ye-2026-world-action-models]], [[papers/jiang-2025-world4rl]]) by quantifying where such models stop generalizing.

## Open questions / things to verify

- No comparison to external ACWM baselines — how would published models (e.g. navigation-trained ones) fare on ACWM-Phys? The "models rely on appearance" claim would be stronger with a model sweep.
- Physics-category vs complexity are partly confounded; a design that holds visual richness fixed while varying DoF would sharpen the conclusion.
- Sim-only: does the InD-strong / OoD-weak pattern transfer to real robot rollouts?
- Would a causal/autoregressive (streaming) variant trade fidelity for the real-time capability the paper says bidirectional diffusion lacks?
- Exact table completeness (per-env full metric grids) should be re-checked against the PDF before citing specific non-headline numbers.

## Connections

- Builds on / relates to (action-conditioned world models as policy/simulator): [[papers/ye-2026-world-action-models]], [[papers/jiang-2025-world4rl]]
- Contrasts with (argues world models + VLA are insufficient for robotics): [[papers/karcini-2026-robots-beyond-vla]]
- Physics-faithfulness cluster (do generative video models respect physics): [[papers/zhao-2026-phyworld]], [[papers/lin-2026-phyground]]
- Adjacent modality (touch-conditioned world models for manipulation): [[papers/higuera-2026-visuo-tactile-world-models]]
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/robotics]]

## Selected quotes

> "Action-conditioned world models (ACWMs) have shown strong promise for video prediction and decision-making. However, existing benchmarks are largely restricted to egocentric navigation or narrow, task-specific robotics datasets, offering only limited coverage of the rich physical interactions required for generalized world understanding." — Abstract

> "existing ACWMs and their accompanying benchmarks suffer from a critical blind spot: *physical diversity*." — Introduction

> "models still rely heavily on appearance statistics rather than internalizing physics." — Conclusion
