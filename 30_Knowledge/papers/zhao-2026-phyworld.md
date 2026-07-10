---
type: paper
title: "PhyWorld: Physics-Faithful World Model for Video Generation"
authors: ["Pu Zhao", "Juyi Lin", "Timothy Rupprecht", "Arash Akbari", "Chence Yang", "Rahul Chowdhury", "Elaheh Motamedi", "Arman Akbari", "Yumei He", "Chen Wang", "Geng Yuan", "Weiwei Chen", "Yanzhi Wang"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2605.19242
rw_id: 01kx5r6a23cxjt0zktkjtrxant
topics: [world-models, video-generation, flow-matching, reinforcement-learning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-10
last_updated: 2026-07-10
---

# PhyWorld: Physics-Faithful World Model for Video Generation

## TL;DR

PhyWorld post-trains a large pretrained image-to-video diffusion model (Wan2.2-I2V-A14B) into a **physically faithful video world model** via two stages: (1) **flow-matching fine-tuning** that adds a video-to-video (V2V) continuation capability for temporally consistent scene continuation, and (2) **Direct Preference Optimization (DPO)** over human physics-preference pairs to push generated dynamics toward physical plausibility. The physics signal comes from a dedicated **250-prompt text/image-to-video (TI2V) physics benchmark** scored on a 1–5 Likert scale by an open-weight 9B video-language judge, so evaluation is reproducible and immune to closed-API drift. PhyWorld reaches **0.769 average on VBench** (vs ≤0.756 for SOTA baselines like Wan2.2-I2V-A14B at 0.756, Cosmos-14B at 0.753) and **3.09 Overall on the physics benchmark** (vs 2.99 for its own frozen base and lower for five other open SOTA models). The gains concentrate exactly on the axes the DPO reward targets — physical-temporal validity (+0.10), persistence (+0.15), and optical physics (+0.21) over the base. The takeaway: continuation + physics-preference post-training turns a general video generator into a more usable world simulator without retraining from scratch.

## Context & motivation

The paper frames video generation models (Sora, CogVideoX, Veo, Cosmos, Wan) as a promising substrate for **world simulators** for Physical AI — embodied agents that need cheap, safe, visually rich practice environments before real-world deployment. Unlike classical physics engines that need hand-authored assets and rules per domain, video models learn rich visual priors about motion, collision, deformation, and lighting from internet-scale video, and can synthesize continuations from text, image, or preceding clips.

Two failures block their use as simulators. **(1) Physical consistency is hard to maintain across generated frames** — SOTA models exhibit background color drift and inconsistent object motion speeds; text-to-video and image-to-video conditioning don't provide enough grounding to infer fine-grained physical attributes. **(2) Physical-law enforcement is architecturally absent** — models are trained on empirical video with no explicit supervisory signal or loss for physical principles, so they lack a mechanism to internalize dynamics/causality and routinely violate constraints. PhyWorld's stated contributions are exactly the two-part fix: a V2V training pipeline for temporal coherence, and DPO-based physics enforcement plus a per-law benchmark. The paper also critiques prior physics benchmarks (PHYSICS-IQ, PHYGENBENCH, VIDEOPHY-2, WORLDMODELBENCH) for relying on implicit prompts, holistic/binary scoring that hides per-law violations, and closed-source evaluators subject to silent drift — motivating its own open, per-law judge.

## Method

Overview (Figure 1): first fine-tune with flow matching to enhance physical consistency; then apply DPO with explicit preference signals to enforce physics.

### Problem formulation
Given an observed scene (a short conditioning clip and/or a first frame), generate a **physically faithful continuation** — a video that preserves the physical state implied by the input and evolves consistently with basic physical principles. Stage 1 optimizes a flow-matching velocity-prediction objective for high-quality V2V continuation; stage 2 optimizes a preference objective that raises the model's relative preference for physically plausible completions.

### Core idea
Treat physical-law violations in a pretrained I2V diffusion model as a **preference-learning problem** and correct them with offline RL (DPO) over human-rated preference pairs, on top of a flow-matching-tuned V2V base — rather than trying to bake physics into the architecture or loss from scratch.

### Architecture / algorithm

**Video-to-video conditioning.** The conditioning video $I \in \mathbb{R}^{C \times R \times H \times W}$ ($R$ frames) is concatenated with zero-padded frames along the temporal axis into a guidance tensor $I_c \in \mathbb{R}^{C \times T \times H \times W}$, then compressed by the **Wan-VAE encoder** into a condition latent $z_c \in \mathbb{R}^{c \times t \times h \times w}$, where $c=16$ latent channels, $t = 1 + (T-1)/4$, $h = H/8$, $w = W/8$ (4× temporal, 8× spatial compression). A **binary mask** $M \in \{0,1\}^{1 \times T \times h \times w}$ marks which frames are retained from the conditioning input (1) versus synthesized (0); $M$ uses the target video's temporal length and is rearranged to $s \times t \times h \times w$ where $s$ is the Wan-VAE temporal stride. The noise latent $z_t$, condition latent $z_c$, and rearranged mask $m$ are concatenated along the channel axis and fed to the **Wan DiT** denoiser.

**Global semantic conditioning.** A **CLIP encoder** extracts features from the *final frame* of the conditioning video; a 3-layer MLP projects them into the model's feature space, producing a global context embedding injected via a **decoupled cross-attention** mechanism. Together the mask (local boundary between given/generated frames) and CLIP embedding (global semantics) give structured guidance for faithful continuation.

**Flow-matching training (Stage 1).** Using the flow-matching / Rectified Flow framework: given a data latent $x_1$, Gaussian noise $x_0 \sim \mathcal{N}(0, I)$, and timestep $t \in [0,1]$ from a **logit-normal** distribution, the intermediate latent is a linear interpolation on the probability-flow path,
$$x_t = t\,x_1 + (1-t)\,x_0 \tag{1}$$
The ground-truth velocity is the path's time derivative,
$$v_t = \frac{dx_t}{dt} = x_1 - x_0 \tag{2}$$
and the model $u(\cdot)$ is trained to regress it with an MSE objective,
$$\mathcal{L} = \mathbb{E}_{x_0, x_1, c_{txt}, t;\,\theta}\; \big\| u(x_t, c_{txt}, t; \theta) - v_t \big\|^2 \tag{3}$$
where $c_{txt}$ is the umT5 text embedding sequence (512 tokens), $\theta$ the model weights, and $u(x_t, c_{txt}, t;\theta)$ the predicted velocity. Training uses a **progressive resolution/duration curriculum** — start on low-resolution video, then scale spatial resolution and temporal length — so the model builds low-level priors before full spatiotemporal complexity.

**Physics enforcement via DPO (Stage 2).** The stage-1 model is the base denoiser; the **policy** $\pi$ adds a low-rank **LoRA** increment $\phi$ (attention + FFN projections; rank $r=16$, scale $\alpha=16$) on top of frozen base weights, and is optimized against a fixed **reference** $\text{ref}$ that copies the same frozen weights with $\phi=0$. For a winner/loser pair $(w,l)$ sharing text prompt $c$ and I2V first frame $x_0$, let $\text{MSE}_{\theta,v}$ be the noise-prediction MSE of denoiser $\theta$ on video $v$ at a timestep $t \sim \mathcal{U}[t_{min}, t_{max}]$ with a **shared** Gaussian noise $\epsilon \sim \mathcal{N}(0,I)$ across $w$ and $l$ (paired-noise variance reduction). The **diffusion DPO loss** is
$$\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta \Delta), \qquad \Delta = (\text{MSE}_{\pi,l} - \text{MSE}_{\pi,w}) - (\text{MSE}_{\text{ref},l} - \text{MSE}_{\text{ref},w}) \tag{4}$$
where $\sigma$ is the logistic sigmoid, $\beta>0$ the DPO inverse temperature (preference strength), and $\Delta$ the **implicit reward margin** — how much more $\pi$ (relative to $\text{ref}$) prefers the winner $w$ over loser $l$ in noise-prediction error. The reference branch is computed gradient-free by zeroing $\phi$ in place (no second model instantiated), halving activation memory at the cost of one extra forward per pair. Training is restricted to a **high-noise window** $t \in [901, 999]$ to suppress reward-hacking via timestep selection and to match where the judge's discriminative signal concentrates.

### Derivations / why it works
The flow-matching loss (3) is justified as a maximum-likelihood-equivalent objective for a continuous-time generative ODE: rather than score matching, it directly regresses the constant velocity $v_t = x_1 - x_0$ of the straight interpolation path (1)–(2), which makes the target trivially closed-form and the training stable. For DPO, the $-\log\sigma(\beta\Delta)$ form is the standard Bradley–Terry preference likelihood transported to diffusion by substituting per-example negative noise-prediction error as the (log-)likelihood proxy; the **reference-subtracted** margin $\Delta$ is what keeps the update from simply lowering loss on all videos — it only rewards moving *winners* below *losers relative to the frozen base*, so the physics-correcting signal doesn't perturb the base generation prior. The paired shared-noise trick reduces variance in $\Delta$ because both branches see the same $\epsilon$.

### Training procedure
- **Stage 1 (flow matching):** start from **Wan2.2-I2V-A14B** (two DiTs for different timestep ranges, both fine-tuned sequentially) with the Diffsynth-Studio framework; learning rate **1e-6**; V2V pipeline uses **17 input frames → 49 ground-truth frames**. Data: **OpenVid-1M**, filtered by (a) CLIP inter-frame cosine similarity to drop near-static and flickering clips, and (b) **UniMatch** optical-flow motion score to drop high-speed/erratic clips — keeping smooth, controlled motion.
- **Stage 2 (DPO):** AdamW, lr **1e-5**, world_size=4, micro-batch 1 per FSDP rank, gradient accumulation 2 (effective batch 8), $\beta=100$; 1,000 pairs = **125 optimizer steps/epoch**, trained **2 epochs (250 steps total)**, final-epoch checkpoint; **16× NVIDIA H100**. $\beta$ is the only tuned hyperparameter, swept over $\{30,100,300\}$ and locked at $\beta=100$ before unblinding (at step 250, $\beta=100$ wins the pre-registered Spearman-monotonicity + step-250-value criterion: Spearman +0.520 vs +0.020 for $\beta=30$; final $\Delta=+0.200$ vs +0.075).

**Preference-data funnel (T0→T3).** Raw signal is the human annotation pool from the TI2V physics benchmark: 2,000 pre-rated videos (250 prompts × 8 generators), each scored 1–5 on semantic alignment (SA), physical-temporal validity (PTV), and persistence, plus applicable physics laws, from ~350 raters yielding ~4,500 cleaned annotations. T0 fixes a per-video aggregate quality score $s(v) = \frac{1}{|R(v)|}\sum_{r \in R(v)} (\text{SA}_r(v) + \text{PTV}_r(v) + \text{persistence}_r(v))$ on a 3–15 scale. T1 forms within-group ordered pairs $(v_w, v_l)$, admitted only if the margin $s(v_w) - s(v_l) \ge 1.0$ and each video has $r_{min}=2$ raters → 3,324 pairs over 250 (prompt, physical-law) groups, with a prompt-disjoint 0.7/0.15/0.15 split; a 42-prompt heldout slice (579 pairs) is the sole eval set. T2 pre-encodes I2V conditioning images and SHA-256-stamps the bytes → 2,202 pairs. T3 samples a **1,000-pair, class-balanced round-4 subset** over seven physical-event classes (collision/rebound, destruction/deformation, fluids, shadow/reflection, chain, rolling/sliding, throwing/ballistic).

### Inference / sampling
At test time PhyWorld runs its V2V (or I2V) diffusion pipeline with the LoRA-augmented policy; the DPO adapter is a small additive weight increment, so inference cost matches the base Wan2.2 pipeline. Videos are generated at 480p for evaluation.

## Experimental setup
- **Datasets / prompts:** VBench (500 random prompts) for generation quality; the authors' **TI2V physics benchmark** (250 prompts, each with a conditioning first frame, organized under a physical-law taxonomy) for physical faithfulness.
- **Baselines:** Wan2.2-I2V-A14B (its own frozen base), Cosmos-14B, LTX-2.3-22B, OmniWeaving, Cosmos-2-2B (VBench); plus Wan2.2-TI2V-5B and LTX-2-19B on the physics benchmark. All run through the V2V pipeline (image-to-video models adapted to V2V).
- **Metrics:** VBench sub-scores (subject/background consistency, motion smoothness, dynamic degree, aesthetic & imaging quality); physics judge scores for SA/PTV/persistence (general) and Solid-Body/Fluid/Optical (physics), with $\text{Overall} = 0.5\cdot\text{mean(SA, PTV, Persist.)} + 0.5\cdot$ pooled mean over (video, law) units.
- **Judge:** an open-weight **Qwen3.5-9B** video-language model fine-tuned end-to-end on the human Likert study; queries **one dimension per inference call** (so per-law scores stay independent) with deterministic greedy decoding for reproducibility.

## Key results
- **VBench (Table 3):** PhyWorld **0.769 avg**, beating Wan2.2-I2V-A14B (0.756), Cosmos-14B (0.753), LTX-2.3-22B (0.751), OmniWeaving (0.750), Cosmos-2-2B (0.739). It leads on subject consistency (0.932), background consistency (0.944), and motion smoothness (0.986).
- **Physics benchmark (Table 4):** PhyWorld **3.09 Overall** vs 2.99 for its frozen base and ≤2.80 for the other five models. Improvements over base concentrate on **PTV +0.10, persistence +0.15, optical +0.21** — the axes the DPO score and high-noise window are designed to move. Appendix D shows four qualitative cases (rigid-body, gravity, fluid continuity, material breakage) where the base violates physics and the RL-DPO LoRA visibly fixes it.

## Ablations
The $\beta$ sweep ($\{30,100,300\}$) functions as the main design ablation: $\beta=100$ dominates on the pre-registered trajectory criterion (Spearman +0.520 vs +0.020 at $\beta=30$). The restriction to the high-noise timestep window $t\in[901,999]$ is motivated as an anti-reward-hacking measure aligned with the judge's discriminative region. A dedicated component-wise ablation table (V2V vs +CLIP vs +DPO isolated) is _not addressed by the source_ in the main text beyond these.

## Limitations
- **Paper's own:** PhyWorld is post-trained from Wan2.2-I2V-A14B and **inherits its data biases and failure modes**; transfer of policies trained *inside* PhyWorld (i.e., using it as a simulator to train downstream agents) is left to future work.
- **Honest reader's view [analyst's]:** the headline margins are small in absolute terms (VBench 0.769 vs 0.756; physics 3.09 vs 2.99) — "non-marginal" is the paper's framing but these are ~1–3% gaps; the physics judge is itself a 9B model fine-tuned on the same annotation pool that also produced the DPO preferences, so judge and reward share a data source (some circularity risk); and the 1,000-pair, 250-step DPO run is small, raising questions about how far the physics gains scale.

## Why it matters [analyst's view]
The interesting move here is **reframing "make video models obey physics" as offline preference optimization** rather than as an architectural or physics-loss problem — and doing it cheaply (a rank-16 LoRA, 250 steps, 16 GPUs) on top of a frozen general-purpose video model. That's a pragmatic recipe that sidesteps the usual "bolt a differentiable simulator onto the model" approach. It sits directly next to [[papers/yuan-2026-physics-alignment]] (which instead does *inference-time* physics alignment with a latent world model — a training-free counterpart to PhyWorld's train-time DPO) and [[papers/esmati-2026-invisible-hand-physics]] (which argues video diffusion models already *encode* physics internally even when outputs violate it — PhyWorld's DPO can be read as surfacing that latent knowledge into the output distribution). It also shares the Wan I2V backbone lineage with [[papers/ye-2026-world-action-models]] (DreamZero, built on Wan2.1-I2V-14B), suggesting a growing "post-train the Wan video model into a world model" pattern. The bigger open question the vault should track: does preference-based physics correction actually generalize the *laws*, or just the seven benchmark event classes it was trained on?

## Open questions / things to verify
- Does the DPO physics gain generalize to held-out physical phenomena outside the seven trained event classes, or is it benchmark-specific?
- How much of the improvement is the V2V flow-matching stage vs the DPO stage? (No clean stage-wise ablation in the main text.)
- Judge/reward share an annotation source — is there measurable reward-hacking of the judge despite the high-noise-window mitigation?
- The relationship to [[papers/lin-2026-phyground]] (same group's benchmark paper): is PhyWorld's "250-prompt TI2V physics benchmark" the same artifact as PhyGround, or a subset/precursor? (See that note.)

## Connections
- Builds on: Wan2.2-I2V-A14B (Wan video foundation model), Rectified Flow / flow matching, Diffusion-DPO.
- Contrasts with: [[papers/yuan-2026-physics-alignment]] (inference-time physics alignment, training-free vs PhyWorld's DPO training)
- Related: [[papers/esmati-2026-invisible-hand-physics]], [[papers/ye-2026-world-action-models]], [[papers/porcher-2026-flowwm]], [[papers/jiang-2025-world4rl]] (diffusion world model + RL for policy refinement), [[papers/joseph-2026-physics-video-world-models]], [[papers/lin-2026-phyground]] (sibling benchmark, same group), [[papers/kerssies-2026-delta-tokens]] (sibling in triage batch)
- Physics-video siblings (ingested via /paper-search 2026-07-10): [[papers/gu-2025-phyworldbench]] (the benchmark PhyWorld's benchmark-adjacent work reacts to), [[papers/meng-2024-phygenbench]] (seminal precursor benchmark), [[papers/xiong-2026-physalign]] (**method sibling** — physics coherence via 3D + feature alignment instead of DPO), [[papers/begiristain-2026-cronos]] (counterfactual-consistency benchmark), [[papers/cao-2026-judgefit]] (per-VLM judge taxonomy), [[papers/xue-2026-acwm-phys]] (action-conditioned video world-model physics)
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/flow-matching]], [[topics/reinforcement-learning]]
- Author indices: [[authors/pu-zhao]], [[authors/juyi-lin]], [[authors/yanzhi-wang]]

## Selected quotes

> "We treat physical law violations ... in a pretrained image-to-video (I2V) diffusion model as a preference learning problem and correct them with offline reinforcement learning (RL), specifically direct preference optimization (DPO), over preference pairs." — §3.2

> "We restrict training to the high-noise window t ∈ [901, 999] to suppress reward-hacking via timestep selection, matching where the automated judge places most of its discriminative signal." — §3.2
