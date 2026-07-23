---
type: paper
title: "DriftWorld: Fast World Modeling through Drifting"
authors: ["Susie Lu", "Haonan Chen", "Weirui Ye", "Yilun Du"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2607.15065
rw_id: 01ky7gfgtz7j82g06c5fa948ya
topics: [world-models, video-generation, generative-models, robotics]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

> **Source note:** Readwise's document export returned empty `html_content` for this item (PDF not yet parsed), so the full text was fetched directly from the arXiv PDF (2607.15065v1) and extracted with pdftotext. Main text and appendices were readable; several figure labels were garbled by the PDF encoding but all tables and equations survived.

## TL;DR

DriftWorld is an action-conditioned world model for robot manipulation that generates future video frames in a **single forward pass**, built on drifting generative models (Deng et al., 2026) instead of diffusion. Rather than iteratively denoising at inference, the model learns during training a *drifting field* — a kernelized attraction–repulsion vector field that pulls generated samples toward the ground-truth future chunk and away from the model's own generations — and is optimized by fixed-point iteration toward the field's equilibrium. This makes rollouts 17× faster on average than diffusion world-model baselines (30+ fps, e.g. 0.0037 s/frame on Push-T vs 1.77 s/frame for Ctrl-World) while matching or beating them on SSIM/PSNR/LPIPS/FID/FVD across Bridge-V2, RT-1, Language Table, Push-T, and Robomimic. The speed directly pays off downstream: ranking K=50 action proposals per step with GPC-RANK lifts Push-T IoU from 0.635 to 0.781 (vs 0.698 for the GPC diffusion world model, in ~2.5× less time), and as an offline policy-evaluation simulator it reaches Pearson correlations of 0.9515 / 0.9916 / 0.9250 with ground-truth policy performance on Push-T / Robomimic Lift / Can.

## Context & motivation

Predictive world models let robots plan by imagining outcomes of candidate actions, and let practitioners rank policies offline without hardware rollouts. But state-of-the-art action-conditioned world models (IRASim, Ctrl-World, GPC's world model, Cosmos-style video models) are almost exclusively multi-step diffusion or autoregressive transformers: tens of forward passes per frame. The paper cites generative predictive control work [ref 5, Qi et al.] reporting that diffusion world-model rollouts consume **90–95% of runtime**, yielding 3+ seconds per decision cycle — which makes evaluating hundreds of action proposals per control step impractical.

Existing fast-sampling routes (progressive distillation, consistency models, rectified flow, adversarial distillation like ADD) mostly assume a pretrained multi-step diffusion teacher. Drifting generative models [ref 6, Deng, Li, Li, Du, He, arXiv 2602.04770] instead train a **one-step generator from scratch** by moving the model's pushforward distribution toward the data distribution via a kernelized attraction–repulsion field. The original drifting formulation was for class-conditional image generation; DriftWorld is claimed as the first adaptation of drifting to conditional sequence prediction. Three adaptations are required: (1) a conditional drifting field defined by observation history and action sequence, with a modification that accentuates action following; (2) a drifting *feature space* using DINOv2/v3 to keep complex scenes sharp; (3) a U-Net that conditions each generated frame precisely on its corresponding action.

## Method

### Problem formulation

Action-conditioned video generation. Let $o_t$ be the visual observation (single- or multi-view image) at time $t$, and $a_t$ the robot action at time $t$. Given a history $o_{t-F:t} = (o_{t-F}, \dots, o_t)$ of $F{+}1$ past frames and a proposed future action sequence $a_{t:t+T} = (a_t, \dots, a_{t+T})$, predict the resulting future observations:

$$o_{t+1:t+T+1} \sim \mathcal{W}(\cdot \mid o_{t-F:t},\; a_{t:t+T})$$

where $\mathcal{W}$ is the action-conditioned world model. Depending on how many conditioning actions are given, the model does single-step ($T{=}1$) or chunk-level simulation.

### Core idea

Replace iterative inference-time denoising with a training-time process: learn a one-step generator whose output distribution is *drifted* toward the data distribution by a contrastive vector field, so that at inference a single forward pass from noise suffices.

### The drifting mechanism

Let $f_\theta$ be the generator, mapping a noise prior $\epsilon \sim p_\epsilon$ (Gaussian) and conditioning $c = (o_{t-F:t}, a_{t:t+T})$ to a generated future chunk $x = f_\theta(\epsilon, c)$. The distribution of generated chunks is the pushforward $q = f_\#\, p_\epsilon$. (Conditioning is suppressed in notation below.)

Define a **drifting field** $V_{p,q}(x)$ that specifies how a sample $x$ must move so that $q$ evolves toward the true data distribution $p$. Given a generated chunk $x_i$ from the current pushforward $q_i$, the drifted chunk is

$$x_{i+1} = x_i + V_{p,q_i}(x_i), \qquad V_{p,q_i}(x) = V^+_p(x) - V^-_{q_i}(x)$$

where $V^+_p(x)$ is the mean-shift vector toward **positive** samples (real data) and $V^-_{q_i}(x)$ is the mean-shift vector toward **negative** samples (the model's own generations) — so the field *attracts* toward data and *repulses* from current generations. Each mean-shift vector is a kernel-weighted average of the differences between the positive/negative samples and $x$. Crucially, when $q_i = p$ the attraction and repulsion cancel, the field is zero, and training is at equilibrium — this is what makes matching the data distribution the fixed point.

DriftWorld's conditional twist: unlike class-conditional drifting, which uses many positives per class, here there is **exactly one correct future** given the history and actions. So:

- the positive $y^+$ is the single ground-truth chunk $o_{t+1:t+T+1}$ from the dataset;
- the negatives $y^-$ are $N_{\text{neg}}$ chunks $\hat{o}_{t+1:t+T+1}$ generated by the model itself (Algorithm 1 also concatenates the last observed frame $o_t$ into the negative set).

### Training objective

$$\mathcal{L} = \mathbb{E}_{\epsilon}\left[\, \big\| f_\theta(\epsilon) - \text{stopgrad}\big(f_\theta(\epsilon) + V_{p,q_\theta}(f_\theta(\epsilon))\big) \big\|^2 \,\right]$$

i.e. **fixed-point iteration**: the current prediction regresses onto a frozen target that is the prediction itself plus the drift. The stop-gradient means the model only chases the drift step; when the drift vanishes (generated distribution matches data), the loss is minimized. Per batch of $B$ videos, $B$ independent conditional drifting fields are computed (one per unique history/action conditioning), each with 1 positive and $N_{\text{neg}}$ negatives; losses sum over the batch.

Algorithm 1 (one training step): sample noise $e \sim \mathcal{N}$ of shape $[N, T, C, H, W]$; generate $x = f(e, \text{obs}, \text{action})$; form negatives $y_{\text{neg}} = \text{cat}([x, \text{obs}[-1]])$; compute $V$ from $(x, y_{\text{pos}}, y_{\text{neg}})$; set $x_{\text{drifted}} = \text{stopgrad}(x + V)$; loss $= \text{MSE}(x - x_{\text{drifted}})$.

### Kernel, normalization, multi-temperature (appendix D.1)

The mean-shift vectors use the kernel $k(x, y) = \exp\!\big(-\tau \cdot \tfrac{1}{\sqrt{C}} \| \tilde{x} - \tilde{y} \| \big)$, where $\tau$ is a temperature and $\tilde{x}, \tilde{y}$ are samples normalized by their average pairwise distance (so the kernel's range is insensitive to feature dimensionality/magnitude). The drifting field is also normalized by its magnitude so multiple fields can be summed. Fields are aggregated over multiple temperatures, $\tilde{V} = \sum_\tau \tilde{V}_\tau$: low $\tau$ makes a sample attracted/repulsed mainly by nearest neighbors, high $\tau$ widens the radius of influence. Temperatures: $\{0.02, 0.05, 0.2\}$ (Push-T, Robomimic), $\{0.02, 0.05\}$ (real-world datasets).

### Accentuating action following

To stop the model from ignoring actions, the negative distribution is optionally modified to a mixture:

$$\tilde{q}(\cdot \mid a_t, o_{t-F:t}) \triangleq (1-\gamma)\, q_\theta(\cdot \mid a_t, o_{t-F:t}) + \gamma\, p(\cdot \mid \varnothing, o_{t-F:t}), \qquad \gamma \in [0,1)$$

i.e. negatives are drawn partly from *real frames where no action was taken* (the ground-truth current frame $o_t$). Repulsing from the "nothing happened" outcome explicitly penalizes the failure mode where the model copies the past frame. An inference-time knob: the U-Net takes an accentuation scale $\alpha$ as input, and during training $\alpha$ is sampled log-uniformly on $[1,4]$ ($p(\alpha) \propto \alpha^{-1}$), so at test time $\alpha$ can be varied without retraining — higher $\alpha$ (2.5–3.5) yields visibly sharper adherence to fast gripper motions than $\alpha \in \{1.0, 2.0\}$ (appendix C.2).

### Drifting space: pixels vs DINO features

The space in which the drifting loss is computed adapts to dataset complexity. Simple simulated data (Push-T, Robomimic): raw pixel space. Complex real-world data (Bridge-V2, RT-1, Language Table): the generator operates in a Stable Diffusion 3 VAE latent space, and the drifting loss is computed in **DINOv2/v3 feature space** via an encoder $\phi$, encouraging $\phi(f_\theta(\epsilon))$ to drift toward $\phi(y^+)$ and away from $\phi(y^-)$. Rationale: the kernel depends on pairwise sample similarity, and DINO features give a semantically meaningful distance. Implementation (appendix): features from blocks 2, 5, 8 of DINOv3 ViT-B/16 → a $16{\times}16$ grid of 768-d tokens → **one drifting field per spatial location** (256 separate losses, weighted mean); plus a parallel drifting loss in the $32{\times}32{\times}16$ VAE latent grid (1024 locations); the two components are summed. The feature extractor is training-time only — zero inference cost.

### Motion weighting

With uniform spatial weights, real robot data traps the model in an action-agnostic local minimum: gripper motion between frames is tiny, so copying $o_t$ (identity mapping) safely minimizes background reconstruction. Fix: weight each spatial location's drifting loss by motion. Overall loss $\mathcal{L} = \mathbb{E}_{h,w}[\, c_{h,w} \cdot \ell_{h,w} \,]$ over the $H{\times}W$ locations, with

$$c_{h,w} = 1 + \lambda \tanh(\alpha \cdot n_{h,w})$$

where $n_{h,w}$ is the normalized feature-space $L_2$ difference between the target future frame(s) and the preceding frame $o_t$ at that location, and $\lambda, \alpha$ are scalars ($c_{h,w}=1$ suffices for simple backgrounds). Moving regions (the gripper) thus dominate the loss.

### Architecture

Action-conditioned U-Net (8.73M / 74.2M / 160M parameters for Push-T / Robomimic / real-world). Inputs: history frames, actions $a_{t:t+T}$, optionally a language instruction. To emit $T$ frames at once it uses a **factorized spatial-temporal convolution** (from AVDC): spatial conv applied identically per timestep, then temporal conv identically per spatial location. Actions condition via **frame-wise FiLM**, so action $a_{t+i}$ modulates exactly the frame $o_{t+i+1}$ it causes. History frames are concatenated channel-wise with the initial Gaussian noise.

### Training procedure

Two-stage **self-forcing** training on complex datasets: stage 1 conditions on ground-truth history; stage 2 initializes from stage 1 and trains autoregressively on the model's own (gradient-detached) generations, each of the $N_{\text{neg}}$ negatives continuing its own rollout — closing the train/test gap for autoregressive rollouts. Optimizer: AdamW ($\beta_1{=}0.9, \beta_2{=}0.95$), lr $1.25\text{e-}5$ (Push-T) / $2.5\text{e-}5$ (others), weight decay 0.01, 500 warmup steps, grad clip 2.0, EMA 0.999. Negatives $N_{\text{neg}}$: 8 / 32 / 64 for Push-T / Robomimic / real-world. Conditioning: current frame + 3 history frames; prediction horizon $T$ = 4 / 2 / 1 / 1 / 1 for Push-T / Robomimic / Bridge-V2 / RT-1 / Language Table.

### Inference / policy simulation

One forward pass: $f_\theta(\epsilon \mid o_{t-F:t}, a_{t:t+T})$, $\epsilon$ Gaussian. For policy rollouts: a vision-based policy $\pi$ emits an action chunk, the world model simulates it, the final predicted observation feeds back into $\pi$ — autoregressive long-horizon simulation entirely in imagination.

## Experimental setup

- **Datasets:** real-world Bridge-V2 (60,096 trajectories, 24 envs, 256×256), RT-1 (87,212 trajectories, 3 envs, 256×256), Language Table (442,226 trajectories, 192×256); simulated Push-T (500 expert + exploration trajectories, 96×96), Robomimic Lift/Can/Square (700 trajectories each incl. failures, 96×96).
- **Baselines:** IRASim, WorldGym, Ctrl-World, the GPC diffusion world model, action-conditioned adaptations of VDM and LVDM, and an MSE baseline (same U-Net, standard MSE loss instead of drifting — the cleanest 1-step control).
- **Metrics:** MSE, SSIM, PSNR, LPIPS, FID, FVD on validation sets; 8-frame autoregressive generation on Bridge-V2/RT-1, full-video autoregressive on the rest. Timing per generated frame on a single H100.

## Key results

*Summarized — see tables for full numbers.*

- **Push-T (Table 1):** 64-frame rollouts — SSIM 0.9925, LPIPS 0.0050, 0.0037 s/frame, vs GPC world model 0.9717 / 0.0239 / 0.0104 and Ctrl-World 0.8914 / 0.1844 / 1.7714. Generation is 3.2–478× faster than baselines. Beats the same-backbone MSE baseline (SSIM 0.9704) at identical speed — isolating the drifting loss as the source of quality.
- **Real-world (Table 3):** Bridge-V2 SSIM 0.821 vs IRASim 0.738 at 0.0300 vs 1.1031 s/frame; RT-1 FVD 68.72 vs IRASim 93.02 at 0.0258 vs 1.1043 s/frame. Language Table FVD 39.09 (best), though IRASim edges PSNR/FID there.
- **Inference-time policy improvement (Table 4):** GPC-RANK with K=50 proposals — Push-T IoU 0.635 → **0.781** (policy 1) and 0.612 → 0.734 (policy 2), beating the GPC world model (0.698/0.614, 2.241 s) and AVDC (0.726/0.682, 106.1 s) at 0.912 s per full proposal set.
- **Offline policy evaluation:** Pearson correlation with ground-truth performance 0.9515 (Push-T, 7 policies; GPC baseline 0.7345 with multiple ranking errors), 0.9916 (Robomimic Lift, 9 policies), 0.9250 (Can), after post-training on failure demonstrations to avoid over-optimistic success estimates.

## Ablations

- **Feature space (Table 5, Bridge-V2):** dropping the DINO extractor (drifting loss in VAE latent space only) collapses FID 0.40 → 34.24 and FVD 13.58/6.20 → 168.34; the gripper turns blurry. DINOv2 vs DINOv3 are close (FVD 13.58 vs 6.20). The single biggest quality lever.
- **Motion weighting + self-forcing (Table 5, RT-1):** FVD 174.67 → 67.88 with motion weighting alone; + self-forcing gives PSNR 22.32 → 23.92 and LPIPS 0.120 → 0.101. Without motion weighting the generated gripper is mostly stationary (appendix C.3).
- **Chunk size (Table 7, Push-T):** 1, 2, or 4 frames per pass all work; 4 is best on SSIM/LPIPS (0.9925/0.0050 at 64 frames).
- **Accentuation scale $\alpha$ (appendix C.2):** higher $\alpha$ at inference sharpens fast gripper motion, tunable without retraining.

## Limitations

Paper-acknowledged: (1) reliance on a robust pretrained feature extractor (DINOv2/v3) for sharpness in complex scenes; (2) training-time memory is higher than diffusion since $N_{\text{neg}}$ (e.g. 64) negative samples must be generated per forward pass, capping the context/generation window under GPU VRAM — future work suggested: sparse history or a temporally-compressing video VAE for longer windows.

[analyst's view] Additional honest flags: DriftWorld is deterministic-target in spirit (single positive per conditioning), so how it handles genuinely stochastic dynamics is untested here; evaluation is manipulation-only with short prediction horizons ($T \le 4$); FID on RT-1 is actually worse than IRASim/LVDM (10.53 vs 4.60/5.40) even though FVD is much better; and Robomimic policy evaluation required post-training on failure data, meaning the out-of-the-box model was over-optimistic.

## Why it matters [analyst's view]

This is the world-model analog of the consistency-model moment for image diffusion, but trained from scratch rather than distilled from a teacher — which matters because robotics world models rarely have a strong pretrained teacher for their exact conditioning. The practical claim is sharp: if rollouts cost milliseconds instead of seconds, inference-time action search (sample K proposals, imagine, rank, execute) becomes the cheap default rather than a luxury — connecting to the vault's inference-time-scaling thread, here applied to control rather than LLMs. Two design choices resonate with other vault notes: computing the generative loss in DINO feature space rather than pixels echoes [[papers/baldassarre-2025-dino-world-models]] (DINO as the backbone of video world models) and [[papers/porcher-2026-flowwm]] (flow matching in feature space for world modeling) — an emerging consensus that *where* you measure generation quality matters as much as the generative mechanism. And the one-pass-per-chunk efficiency angle parallels [[papers/kerssies-2026-delta-tokens]] (compressing frames to single tokens) — different axis (representation compression vs sampling-step compression), same bottleneck. Against [[papers/ding-2024-diffusion-world-model]], which kept multi-step diffusion but amortized rollout error, DriftWorld removes the multi-step machinery entirely. The offline policy-evaluation correlations (up to 0.99) are the quietly important result: if they hold beyond Robomimic-scale tasks, learned simulators become a credible substitute for hardware evaluation.

## Open questions / things to verify

- The drifting framework's convergence properties: the paper states equilibrium at $q = p$ but gives no proof here — chase the parent paper (Deng et al., arXiv 2602.04770, `_needs note_`) for the theory.
- How sensitive is the method to $N_{\text{neg}}$? The 8→64 range across datasets is stated but never ablated in this text.
- Stochastic environments: single-positive drifting presumes one correct future; multimodal dynamics (e.g. contact-rich uncertainty) are untested.
- Whether $\gamma$ (the no-action negative mixture weight) was ablated — the mechanism is described but no table isolates it (only the inference-time $\alpha$ study).
- Scaling: 160M parameters is small next to Cosmos-class video models; does drifting hold up at scale?

## Connections

- Contrasts with: [[papers/ding-2024-diffusion-world-model]] — keeps multi-step diffusion for world modeling; DriftWorld eliminates iterative sampling entirely
- Parallel approach: [[papers/porcher-2026-flowwm]] — feature-space generative world modeling via flow matching; DriftWorld does feature-space drifting
- Shares DINO-feature foundation: [[papers/baldassarre-2025-dino-world-models]]
- Same efficiency goal, different axis: [[papers/kerssies-2026-delta-tokens]] — token compression vs step compression
- Field context: [[papers/ding-2024-world-models-survey]]
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/generative-models]], [[topics/robotics]]
- Author indices: [[authors/yilun-du]] (if tracked)

## Selected quotes

> "recent work on generative predictive control [5] reports that diffusion world model rollouts consume 90–95% of runtime, resulting in 3 or more seconds per decision cycle" — §1

> "Drifting generative models [6] instead learn a one-step generator from scratch by iteratively moving the model's pushforward distribution toward the data distribution using a kernelized attraction-repulsion field." — §2

> "Crucially, equilibrium is reached when the generated distribution qi matches the true conditional video distribution p: at this point, the drifting field reaches 0, so samples no longer move." — §3.2

> "The straightforward approach of weighting the losses from all spatial locations equally can trap the model into learning an action-agnostic local minimum, where it learns an identity mapping (i.e., copying the past observation ot) to safely minimize background reconstruction error." — §3.3
