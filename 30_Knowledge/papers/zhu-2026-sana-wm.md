---
type: paper
title: "SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer"
authors: ["Haoyi Zhu", "Haozhe Liu", "Yuyang Zhao", "Tian Ye", "Junsong Chen", "Jincheng Yu", "Tong He", "Song Han", "Enze Xie"]
year: 2026
venue: arXiv preprint (NVIDIA)
url: https://arxiv.org/abs/2605.15178
rw_id: 01ky5t3xghppd8zy1y95m78cer
topics: [world-models, video-generation, diffusion-models, gpu-optimization]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer

## TL;DR

SANA-WM is a 2.6B-parameter open-source camera-controlled world model from NVIDIA: given a first frame, a text prompt, and a 6-DoF camera trajectory, it generates a one-minute 720p video that follows the trajectory while preserving scene identity. The core trick is a hybrid diffusion-transformer backbone that replaces most softmax attention with *frame-wise Gated DeltaNet* (a linear recurrence with decay and delta-rule correction, scanned one latent frame at a time), keeping minute-scale context memory constant, plus a dual-rate camera-conditioning scheme (latent-rate UCPE attention + raw-frame Plücker raymap mixing) and a second-stage long-video refiner. It is trained natively at one-minute length on only ~213K publicly-sourced clips re-annotated with metric-scale poses, in 15 days on 64 H100s. On the authors' new one-minute benchmark it beats larger open baselines (LingBot-World 14B+14B, HY-WorldPlay 8B, Matrix-Game 3.0 5B) on action-following (RotErr 4.50°/8.34° with refiner vs. 10.47°/18.99° for the best baseline) with comparable VBench quality (80.62/81.89 vs. 81.82/81.89), at up to 36× higher throughput and single-GPU inference; a 4-step distilled NVFP4 variant denoises a 60s 720p clip in 34s on one RTX 5090.

## Context & motivation

World models — generative simulators that roll out observations under actions — are becoming an interface for embodied simulation and interactive environments. Recent open systems (LingBot-World, HY-WorldPlay, Matrix-Game 3.0, Infinite-World) achieve minute-scale action-conditioned rollouts but require large models, large-scale data, long training, and multi-GPU inference. The tempting cheap alternative — distilling long-rollout models from short-video generators — fails because short-horizon teachers provide limited supervision for minute-scale scene persistence and trajectory following. The paper's question: *can a high-fidelity, camera-controllable, one-minute world model be trained natively while keeping data, training, and inference costs accessible?*

The paper's stated contributions: (i) a natively one-minute-trained, 720p, action-controllable world model with accessible cost; (ii) an efficiency-oriented architecture combining high-compression video latents (LTX2 VAE), hybrid GDN/softmax long-context modeling, and dual-branch camera control; (iii) a long-video second-stage refiner; (iv) a robust metric-scale pose annotation pipeline and a one-minute world-model benchmark. It builds directly on SANA-Video (linear-attention video diffusion) and the LTX2 tokenizer.

## Method

### Problem formulation

Input: a first frame (image), a scene-static text prompt, and a continuous metric-scale 6-DoF camera trajectory (camera-to-world pose + intrinsics per frame). Output: a 60-second, 720p, 16 fps video (960 frames; 961 camera frames) that follows the trajectory. The generator is a latent diffusion transformer trained with the standard flow-matching objective on 961-frame clips; three inference variants exist: bidirectional (offline, highest quality), chunk-causal autoregressive (sequential rollout), and a 4-step self-forcing-distilled AR variant for deployment.

### Core idea

Make minute-scale context affordable by doing almost all temporal mixing with a *frame-wise linear recurrence with decay* (Gated DeltaNet scanned per latent frame, state size constant in sequence length), interleaving occasional softmax attention blocks for exact long-range recall — and inject camera control at two rates so aggressive VAE temporal compression doesn't destroy fine motion.

### Architecture / algorithm

**Backbone.** A 20-block DiT ($d_{\text{model}}=2240$, 20 heads, head dim $D=112$): 15 frame-wise GDN blocks interleaved with softmax-attention blocks at layers $\{3,7,11,15,19\}$. Video is tokenized by the LTX2 VAE with $C=128$ latent channels (representation 2.0× smaller than ST-DC-AE, 8.0× smaller than Wan2.1-VAE); each latent frame summarizes 8 raw frames (temporal stride 8). Every block carries the dual camera-conditioning branches.

**From cumulative linear attention to frame-wise GDN.** The starting point, SANA-Video, uses ReLU cumulative linear attention: for latent frame $t$ with $S$ spatial tokens, with per-head queries/keys/values $\mathbf{Q}_t,\mathbf{K}_t,\mathbf{V}_t \in \mathbb{R}^{D\times S}$ and $\phi = \text{ReLU}$,

$$\widetilde{\mathbf{O}}_t^{\text{LA}} = \Big(\underbrace{\textstyle\sum_{\tau=0}^t \mathbf{V}_\tau \phi(\mathbf{K}_\tau)^\top}_{\mathbf{A}_t^{\text{LA}}}\Big)\,\phi(\mathbf{Q}_t) = \big(\mathbf{A}_{t-1}^{\text{LA}} + \mathbf{V}_t\phi(\mathbf{K}_t)^\top\big)\phi(\mathbf{Q}_t),$$

i.e. instead of pairwise attention weights $\mathbf{Q}^\top\mathbf{K}$, key–value outer products are accumulated into a constant-size state $\mathbf{A}_t^{\text{LA}} \in \mathbb{R}^{D\times D}$ updated once per latent frame (numerator only shown; standard normalizer omitted in the paper). The problem: this state has **no decay or saliency mechanism** — stale features accumulate with the same effective weight as recent ones, and at minute scale the unbounded state causes drift and training instability.

Gated DeltaNet (GDN) fixes this with a decay gate and a delta-rule correction. Token-wise GDN:

$$\mathbf{S}_i = \gamma_i \mathbf{S}_{i-1} + (\mathbf{v}_i - \gamma_i \mathbf{S}_{i-1}\hat{\mathbf{k}}_i)\,\beta_i\,\hat{\mathbf{k}}_i^\top, \qquad \mathbf{o}_i = \mathbf{S}_i \hat{\mathbf{q}}_i,$$

where $\mathbf{S}_i \in \mathbb{R}^{D\times D}$ is the recurrent state, $\hat{\mathbf{q}}_i,\hat{\mathbf{k}}_i,\mathbf{v}_i \in \mathbb{R}^D$ are normalized query/key and value, $\beta_i \in [0,1]$ is an update gate and $\gamma_i \in (0,1]$ a decay gate. The term $(\mathbf{v}_i - \gamma_i\mathbf{S}_{i-1}\hat{\mathbf{k}}_i)$ is the *residual* between the target value and what the current state already predicts for key $\hat{\mathbf{k}}_i$ — only that residual is written, while $\gamma$ forgets obsolete content. Standard GDN scans one token per step; SANA-WM scans **one latent frame per step**: with $\hat{\mathbf{Q}}_t,\hat{\mathbf{K}}_t,\mathbf{V}_t \in \mathbb{R}^{D\times S}$, frame-level decay $\gamma_t \in (0,1]$, and per-token update gates $\boldsymbol{\beta}_t = \text{diag}(\beta_{t,1},\dots,\beta_{t,S})$,

$$\mathbf{S}_t = \mathbf{S}_{t-1}\mathbf{M}_t + \mathbf{U}_t, \qquad \mathbf{M}_t = \gamma_t\big(\mathbf{I} - \hat{\mathbf{K}}_t \boldsymbol{\beta}_t \hat{\mathbf{K}}_t^\top\big), \qquad \mathbf{U}_t = \mathbf{V}_t \boldsymbol{\beta}_t \hat{\mathbf{K}}_t^\top, \qquad \mathbf{O}_t = \mathbf{S}_t \hat{\mathbf{Q}}_t,$$

where $\mathbf{S}_t, \mathbf{M}_t, \mathbf{U}_t \in \mathbb{R}^{D\times D}$ are the frame state, transition matrix, and additive update, and $\mathbf{O}_t \in \mathbb{R}^{D\times S}$ the frame's output tokens. The state stays $D\times D$ regardless of video length; one recurrent step consumes all $S$ spatial tokens of a frame.

**Algebraic stabilization (key scaling).** Since $\mathbf{S}$ is repeatedly multiplied by $\mathbf{M}_t$, the transition must be non-expansive. Let $\bar{\mathbf{K}}_t = \text{ReLU}(\text{RMSNorm}(\mathbf{K}_t))$ and $\mathbf{A}_t = \bar{\mathbf{K}}_t\boldsymbol{\beta}_t\bar{\mathbf{K}}_t^\top$. The key energy is $\text{tr}(\mathbf{A}_t) = \sum_{s=1}^S \beta_{t,s}\|\bar{\mathbf{k}}_{t,s}\|_2^2$; because $\mathbf{A}_t$ is PSD, $\lambda_{\max}(\mathbf{A}_t) \le \text{tr}(\mathbf{A}_t)$, and an $O(S)$ trace can make $\mathbf{I}-\mathbf{A}_t$ expansive (spatial explosion). The fix scales *only* the keys:

$$\hat{\mathbf{K}}_t = \bar{\mathbf{K}}_t \cdot \tfrac{1}{\sqrt{D\cdot S}},$$

so with RMS-normalized keys and $\beta_{t,s}\in[0,1]$, $\text{tr}(\hat{\mathbf{K}}_t\boldsymbol{\beta}_t\hat{\mathbf{K}}_t^\top) \le 1$, hence $\|\mathbf{M}_t\|_2 \le \gamma_t \le 1$. The $1/\sqrt{D}$ factor matches token-wise GDN's L2 key normalization; the extra $1/\sqrt{S}$ averages over spatial tokens. Ablation shows this is the *only* stable variant — $\ell_2(1/\sqrt{D})$ and no-scale baselines hit NaNs at steps 16 and 1 respectively.

**Bidirectional and chunk-causal variants.** The same recurrence is used bidirectionally by summing forward and reversed-time scans. For chunk-causal AR inference, latent frames are partitioned into chunks: the forward scan stays global, the reversed scan resets at chunk boundaries — each chunk gets local future context without future leakage.

**Hybrid GDN/softmax.** Every fourth block is standard softmax attention (FlashAttention), fine-tuned in while retaining the original QKV/output projections — these blocks anchor exact long-range recall and long-term spatial consistency that a compressed recurrent state cannot guarantee. For deployment, softmax layers get attention-sink tokens (first latent frame as sink) plus local temporal windows, keeping softmax memory and per-chunk latency constant in rollout length.

### Dual-branch camera control

Aggressive temporal compression (8 raw frames per latent) makes single-rate conditioning insufficient, so control is injected at two rates.

**Coarse branch — ray-local UCPE at latent-frame rate.** For each latent token at frame $t$, spatial cell $s$: with camera-to-world pose $\mathbf{T}^{c2w}=[\mathbf{R}\,|\,\mathbf{o}]$ and intrinsics $\mathbf{A}$, the corresponding pixel is unprojected to a world-space ray (center $\mathbf{o}\in\mathbb{R}^3$, unit direction $\mathbf{d}_{t,s}\in\mathbb{R}^3$). A ray-local basis $\mathbf{z}=\text{norm}(\mathbf{d}_{t,s})$, $\mathbf{x}=\text{norm}(\mathbf{u}\times\mathbf{z})$, $\mathbf{y}=\mathbf{z}\times\mathbf{x}$ ($\mathbf{u}$ = camera vertical axis) defines a homogeneous transform $\mathbf{D}_{t,s}\in\mathbb{R}^{4\times4}$ from world coordinates to the ray's local frame. Each camera-branch attention head splits its channels into geometric channels (transformed by $\mathbf{D}$, applied blockwise to 4-D homogeneous groups) and standard RoPE channels:

$$\widetilde{\mathbf{Q}}_i^c = (\mathbf{D}_i^\top \oplus \text{RoPE}_i)\,\mathbf{Q}_i^c, \quad (\widetilde{\mathbf{K}}_i^c, \widetilde{\mathbf{V}}_i^c) = (\mathbf{D}_i^{-1} \oplus \text{RoPE}_i)(\mathbf{K}_i^c, \mathbf{V}_i^c), \quad \mathbf{O}_i^c = (\mathbf{D}_i \oplus \text{RoPE}_i^{-1})\,\text{GDN}_{\text{cam}}(\widetilde{\mathbf{Q}}^c,\widetilde{\mathbf{K}}^c,\widetilde{\mathbf{V}}^c)_i,$$

where $\oplus$ is block-diagonal composition over the UCPE and RoPE channel groups. Wrapping the recurrence in $\mathbf{D}^{-1}\cdot(\cdot)\cdot\mathbf{D}$ makes attention geometry-aware — interactions happen in a pose-relativized frame, so the model learns *relative* camera geometry rather than absolute coordinates. The camera branch has its own QKV projections but shares the frame-wise GDN gates with the main branch; its zero-initialized output projection is added to the main attention output (preserving the pretrained model at init).

**Fine branch — raw-frame Plücker mixing.** Each latent token summarizes 8 raw frames with distinct poses, which the coarse branch can't see. Per raw frame $f$ and pixel $p$, a Plücker raymap $\boldsymbol{\pi}_{f,p} = (\mathbf{d}_{f,p},\, \mathbf{o}_f \times \mathbf{d}_{f,p}) \in \mathbb{R}^6$ is computed; the 8 raymaps within one VAE temporal stride are packed into a 48-channel tensor, passed through a zero-initialized 3D patch embedder, and added via a zero-initialized per-block projection right after each self-attention output. This restores sub-stride camera motion the latent rate discards.

### Second-stage refiner (truncated-σ flow matching)

A dedicated refiner corrects structural artifacts across the full minute. Trained on pairs $(x_\ell, x_h)$ — stage-1/degraded latent and high-fidelity target. Source point: $x_1 = (1-\sigma_{\text{start}})x_\ell + \sigma_{\text{start}}\epsilon$, $\epsilon \sim \mathcal{N}(0,I)$, with $\sigma_{\text{start}} = 0.909375$. Sample $\sigma_t$ from a shifted-logit-normal truncated to $(0, \sigma_{\text{start}}]$, set $\alpha = \sigma_t/\sigma_{\text{start}}$, interpolate $x_t = (1-\alpha)x_h + \alpha x_1$, and regress the velocity $v^\star = (x_1 - x_h)/\sigma_{\text{start}}$:

$$\mathcal{L}_{\text{refiner}} = \mathbb{E}_{\sigma_t,\epsilon}\,\|v_\theta(x_t, \sigma_t, c) - v^\star\|_2^2,$$

where $c$ = text + camera + reference-image conditioning. The truncation means the model learns *refinement from a partially-noised source*, not full generation from pure noise. A clean slice of $x_h$ is prepended as reference tokens (fixed KV anchors via block-wise mask, excluded from the loss) to preserve stage-1 appearance. Implementation: initialized from the 17B LTX-2 model, rank-384 LoRA on attention + FFN projections under FSDP2. Directly fine-tuning the distilled few-step refiner was unstable, so the LoRA is trained on the multi-step base model and **zero-shot merged** into the distilled model, which then runs only 3 Euler steps (sigmas $[0.909375, 0.725, 0.421875, 0]$).

### Training procedure

Progressive four-stage schedule (all stages AdamW, BF16, grad clip 0.5; effective global batch 64→32 on 64 H100s):
1. **VAE adaptation** (~3.5 days, 50K steps, before DiT stages): swap in LTX2-VAE, re-init patchify + output projection, full-model fine-tune.
2. **Stage 1 — frame-wise GDN** on 5s SANA-Video SFT clips, lr $5\times10^{-5}$, 30K steps (~2.75 days).
3. **Stage 2 — hybrid attention** (softmax every 4th block), 5s clips, 30K steps (~2 days).
4. **Stage 3 — minute-scale + camera control**: 961-frame clips, SANA-WM 213K-clip data, lr $1\times10^{-5}$, 31K steps (~8 days), context-parallel size 2, precomputed VAE latents.
5. **Stage 4 — SFT + chunk-causal fine-tune + self-forcing distillation to 4 steps**: ~50K high-quality clips, 10K steps (~2.5 days).

**Context-parallel training** exploits that the GDN update is affine: rank $p$ holding frame shard $\mathcal{I}_p$ computes a transition composite $\mathbf{C}_p = \prod_{t\in\mathcal{I}_p}\mathbf{M}_t$ and input composite $\mathbf{H}_p$ such that $\mathbf{S}_{\text{end}}^{(p)} = \mathbf{S}_{\text{start}}^{(p)}\mathbf{C}_p + \mathbf{H}_p$; all-gathering these compact summaries and composing an exclusive prefix ($\bar{\mathbf{S}}_{p+1} = \bar{\mathbf{S}}_p\mathbf{C}_p + \mathbf{H}_p$) recovers the mathematically exact initial state per rank with minimal communication. Temporal convolutions use halo exchange ($k-1$ boundary frames). Custom fused Triton kernels (RMSNorm + ReLU + key scaling + UCPE/RoPE prep + GDN scan) give ~1.5–2× efficiency.

**Data** (~213K clips total): SpatialVID-HQ (158K, 10s), MiraData (19K, 60s), DL3DV 3DGS-rendered (14.9K synthetic 60s), Sekai Walking-HQ (9.8K, 60s), DL3DV (5.7K), Sekai Game (3.6K), OmniWorld (1.7K). Pose annotation: modified VIPE with depth backend replaced by Pi3X (long-sequence-consistent structure) fused with MoGe-2 (metric-scale anchor; per-frame scale solved with inverse-depth weights, EMA-smoothed), plus per-frame intrinsics optimization in bundle adjustment. DL3DV static scenes are augmented by fitting FCGS 3D Gaussian Splats, rendering 40 candidate one-minute trajectories per scene (orbit/spiral/dolly/fly-through/etc. with coverage checks) and cleaning with DiFix3D. Captions are deliberately *scene-static* (no "pan left") so trajectory supervision flows only through the pose branch, not the text.

### Inference / sampling

Three single-GPU variants: bidirectional (best quality), chunk-causal AR (60 denoising steps), and 4-step distilled AR with attention sinks. Efficiency path for a 60s 720p clip: H100 60-step AR = 22.1 min → 4-step distilled = 69s → + attention sink = 55s → RTX 5090 4-step + sink = 39s → + NVFP4 quantization = 34s (~39× overall; DiT itself 21.7 min → 8s). Without the sink, the 5090 OOMs.

## Experimental setup

- **Benchmark (new, released by the authors)**: 80 first-frame images at 1280×720 generated by Nano Banana Pro across game / indoor / outdoor-city / outdoor-nature (20 each); Simple (smooth arcs, S-curves, figure-eights) and Hard (whip-pans, vertical motion, double loops) trajectory splits, 60s at 16 fps, Catmull-Rom + quaternion Squad interpolated with collision checks against Pi3X point clouds; stored revisit pairs (<0.5 m, <20° apart) for loop-closure memory evaluation.
- **Baselines**: Infinite-World (1.3B, 480p), LingBot-World (14B+14B, 480p), HY-WorldPlay (8B, 480p), Matrix-Game 3.0 (5B, 720p) — each in its multi-step undistilled AR setting via its native control interface.
- **Metrics**: VBench (custom-input mode, 8 dimensions + Overall); pose accuracy via Pi3X-recovered trajectories, Umeyama Sim(3)-aligned — RotErr (deg), TransErr, CamMC (Frobenius pose distance); revisit memory PSNR/SSIM/LPIPS; temporal degradation ΔIQ (first 10s window imaging quality minus last window); peak memory and videos/hour on 8 H100s.

## Key results

- **Action following (best in class)**: SANA-WM + refiner gets RotErr 4.50°/8.34° (Simple/Hard) and CamMC 1.41/1.44 vs. best baseline LingBot-World at 10.47°/18.99° — while being 2.6B vs. 14B+14B and 720p vs. 480p.
- **Visual quality (comparable)**: VBench Overall 80.62/81.89 with refiner, vs. LingBot-World 81.82/81.89.
- **Efficiency**: 51.1 GB peak, 24.1 videos/hour (74.7 GB / 22.0 with refiner, still within one 80 GB H100) vs. LingBot-World's 454.1 GB / 0.6 videos/hour on 8 GPUs — the headline ~36× throughput. All-softmax baseline OOMs at 60s; recurrent variants scale compactly.
- **Long-horizon stability**: refiner cuts ΔIQ from 3.79/3.09 to 1.17/0.31 (HY-WorldPlay degrades by 23.59/25.88); revisit PSNR 14.46/14.80 dB (first on Hard).
- **Bidirectional vs. AR** (App. E): bidirectional variant is notably better at control (RotErr 3.11°/3.17° vs. 7.59°/10.02° AR) at slightly lower dynamic degree — the AR gap is the price of causal rollout.

## Ablations

- **Progressive-training** (5s, VBench-I2V): LTX2 VAE swap is quality-neutral (+0.0012 Total) but cuts memory 8.9→5.4 GB and latency 3.4×; hybrid GDN+softmax raises Total 0.8390→0.8530 at ~5.7 GB. Efficiency comes from the tokenizer; quality comes from the hybrid backbone.
- **Key scaling is load-bearing**: under identical all-GDN training, $1/\sqrt{D\cdot S}$ is the only stable variant; $\ell_2(1/\sqrt{D})$ and no-scale NaN at steps 16 and 1.
- **Camera conditioning** (OmniWorld, 5s): no control RotErr 16.93 → Plücker-only 16.02 (input-level conditioning barely helps) → PRoPE 6.29 → UCPE-only 7.73 → **UCPE+Plücker 6.21, CamMC 0.2047** (best control, competitive FVD). Attention-level geometric conditioning is what matters; the fine branch adds the last increment.
- **Refiner adaptation matters**: the original short-video LTX-2.3 refiner applied to 60s latents *destroys* quality (Overall 71.37, dynamic degree 0.0, RotErr degrades to 8.65/27.38) vs. the adapted long-video refiner (80.62/81.89) — a generic second-stage decoder is not sufficient at minute scale.

## Limitations

Paper's own: remains scale-limited; lacks explicit 3D scene memory; can drift in dynamic scenes, rare viewpoints, or longer rollouts; future work should scale model/data, explore robot-action or point-tracking controls, persistent scene memory, and real-time streaming refiners. Honest-reader additions [analyst's view]: revisit PSNR ~14.5 dB is objectively low for loop closure — "scene memory" here is soft persistence, not geometric consistency; the benchmark is self-built and first frames are model-generated (Nano Banana Pro), so no real-world ground-truth video exists for the evaluated scenes; pose accuracy is measured with the same estimator family (Pi3X) used in training annotation, which could correlate errors; the refiner adds a 17B model to a "2.6B" system, complicating the parameter-count comparison.

## Why it matters [analyst's view]

This is the clearest demonstration yet in the vault that the linear-attention/SSM hybrid recipe from long-context LLMs transfers to video world models: a $D\times D$ recurrent state per head plus sparse exact-attention anchors is enough for 60s of 720p, where a full softmax KV cache literally OOMs. The frame-wise (rather than token-wise) GDN scan is the interesting adaptation — it treats a latent frame's spatial tokens as one batched recurrent update, and the $1/\sqrt{DS}$ non-expansiveness argument (bounding $\|\mathbf{M}_t\|_2$ via the trace of a PSD matrix) is a nice, reusable piece of stability analysis for anyone batching linear-attention updates. The systems story is equally notable for this vault's GPU interests: affine-recurrence prefix composition for context parallelism, fused Triton scan kernels, attention sinks for constant-memory rollout, and NVFP4 deployment on a consumer 5090 form a complete efficiency stack. The dual-rate conditioning insight — high-compression tokenizers destroy sub-stride action information, so conditioning must be injected at both latent and raw-frame rates — likely generalizes to any action-conditioned model built on aggressive video VAEs. Finally, the scene-static captioning trick (strip camera language from captions so control must flow through the pose branch) is a cheap, transferable idea for disentangling conditioning channels.

## Open questions / things to verify

- How much does the 17B refiner contribute vs. the 2.6B backbone — could a small refiner suffice, or is the quality parity with LingBot-World mostly bought by LTX-2?
- Is Pi3X-based pose evaluation biased toward a model trained on Pi3X-annotated data? An independent pose estimator on the benchmark would settle it.
- The bidirectional→AR control gap (RotErr 3.1°→7.6°) is large; does self-forcing distillation widen or shrink it? (The distilled variant's pose accuracy isn't in the main table.)
- Can the frame-wise GDN state support genuine revisit memory at longer horizons, or is explicit 3D/geometric memory (the paper's own stated gap) unavoidable?
- App. C/E details (per-dimension VBench, kernel specifics) worth a look on deep read; check the GitHub release for the Triton kernels.

## Connections

- Extends: SANA-Video (linear-attention video DiT) and LTX2 (high-compression tokenizer, refiner recipe) — _needs note_ for both.
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/diffusion-models]], [[topics/gpu-optimization]]
- Related in vault:
  - [[papers/kerssies-2026-delta-tokens]] — same efficiency battle (cutting video world-model token/compute cost), different lever.
  - [[papers/cai-2026-mode-mean-seeking]] — long-horizon video generation degradation, the failure mode SANA-WM's ΔIQ metric and refiner target.
  - [[papers/gao-2025-adaworld]] — action-conditioned world modeling with learned actions; SANA-WM's continuous metric 6-DoF camera control is the explicit-action counterpoint.
  - [[papers/baldassarre-2025-dino-world-models]] — representation-centric predictive world models, the non-generative alternative setting the paper contrasts itself with.
  - [[papers/ding-2024-world-models-survey]] — situates the generative-simulator branch SANA-WM belongs to.

## Selected quotes

> "can we natively train a high-fidelity, camera-controllable, one-minute world model while keeping data, training, and inference costs accessible?" — §1

> "This compact state has no explicit decay or saliency mechanism: stale features accumulate with the same effective weight as more recent ones. At the minute scale, the unbounded growing state causes drift and degrades training stability." — §3.2

> "we use scene-static captions that describe objects, layout, and appearance while omitting camera actions such as 'pan left' or 'move forward.' This prevents text from leaking trajectory supervision and forces motion control through the pose branch." — §4

> "The recurrent/linear variants keep compact clean-context states, while all-softmax grows its KV cache and runs out of memory at 60s." — §5.4
