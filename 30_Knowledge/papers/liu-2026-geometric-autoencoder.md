---
type: paper
title: "Geometric Autoencoder for Diffusion Models"
authors: ["Hangyu Liu", "Jianyong Wang", "Yutao Sun"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2603.10365
rw_id: 01kx5vcwamrj5b81x5527rf7sz
topics: [diffusion-models, generative-models, representation-learning, self-supervised-learning]
priority: medium
read_state: queued
relevance: ""
added: 2026-07-14
last_updated: 2026-07-14
---

# Geometric Autoencoder for Diffusion Models

## TL;DR

**Geometric Autoencoder (GAE)** is a principled tokenizer/autoencoder for latent diffusion that fixes three things prior VFM-based latent designs handle heuristically: how to inject Vision-Foundation-Model (VFM) semantics into a *compact* latent, how to constrain the latent distribution, and how to keep reconstruction stable under diffusion noise. Its three moves are (1) **Latent Alignment** — supervise the compressed latent bottleneck directly against a *downsampled* DINOv2 teacher (rather than aligning high-dim encoder features "pre" or projecting latents back up "post"), using a learned "Semantic Teacher" that distills 1024-dim VFM features into a 32/64-dim target; (2) **Latent Normalization** — replace the VAE's KL penalty with a parameter-free RMSNorm that projects the latent mean onto a unit hypersphere; and (3) **Dynamic Noise Sampling** (from σ-VAE) — sample a random noise scale $\sigma$ per step so the decoder learns a continuous, perturbation-robust manifold. On ImageNet-1K 256×256 with a 32-dim latent and a LightningDiT-XL generator, GAE reaches **gFID 1.82 at 80 epochs and 1.31 at 800 epochs without CFG** (1.13 with CFG), beating VA-VAE, REPA-E, FAE and matching/edging RAE, while attaining strong linear-probing accuracy (69.4% at d=32, 78.3% at d=64). The "geometry" is the hyperspherical latent manifold plus bottleneck-level semantic anchoring — not a Riemannian-metric method.

## Context & motivation

Latent diffusion models (LDMs) [Rombach et al. 2022, Vahdat et al. 2021] operate in a VAE-compressed space and underpin most current SOTA generative systems (SD3/Esser et al. 2024, Seedream, Wan). A recent frontier fuses **semantic representation learning** into the latent so diffusion converges faster. The paper groups prior efforts into three trajectories: (i) **VAE + semantic supervision** during autoencoder training (VA-VAE [Yao et al. 2025a], AlignTok [Chen et al. 2025b]); (ii) **alignment losses** that tie diffusion-transformer internal features to VFM feature spaces (REPA [Yu et al. 2024], REPA-E [Leng et al. 2025]); and (iii) **adapting VFMs directly** as latents/tokenizers (RAE [Zheng et al. 2025], FAE [Gao et al. 2025], Bi et al. 2025, Gui et al. 2025).

The stated problem: latent-space design "remains largely heuristic." Three concrete gaps motivate GAE — (1) alignment strategies vary wildly and often yield sub-optimal representations because there is no principled analysis of *where* to align; (2) VFM-derived latents are high-dimensional and inflexible — you can't easily dial the latent dimension down to what diffusion wants (e.g. d=32); (3) frameworks that hit both semantic and dimensional targets (e.g. VTP [Yao et al. 2025b]) still generate poorly because their **reconstruction stability under noise is fragile**. GAE's contribution is to treat these as one coupled design problem — semantic density, compactness, and reconstruction robustness — and address each with a targeted mechanism.

## Method

### Problem formulation
Learn an autoencoder $(E_p, A_p, D_p)$ mapping an image $x$ to a compact latent $z \in \mathbb{R}^{16\times16\times d}$ (with $d = 32$ or $64$; spatial downsampling factor $f=16$) such that: (a) $z$ reconstructs $x$ faithfully, (b) $z$ retains VFM-level semantic discriminability (measured by linear probing, LP), and (c) $z$'s distribution is stable for a downstream latent-diffusion generator (LightningDiT-XL). The autoencoder is trained first; the diffusion model is trained on the frozen latents afterward.

### Core idea
Anchor the *compressed bottleneck itself* to a **dimensionality-matched** semantic teacher (a learned downsampler of DINOv2), and replace the KL prior with a hard geometric constraint (unit-hypersphere RMSNorm) plus stochastic noise-scale sampling — giving a latent that is simultaneously compact, semantically rich, and denoising-robust.

### Architecture / algorithm
Dual-branch design (Fig. 2):

- **Pixel branch (trainable):** Encoder $E_p$ (ViT-L, 24 layers, 16 heads, with RMSNorm + SwiGLU refinements) extracts spatial features; a linear **Projector** $A_p$ maps $16\times16\times1024 \to 16\times16\times d$; **Decoder** $D_p$ reconstructs $16\times16\times3$. ViT-based AE is chosen over conv-nets for throughput/scalability [Teng et al. 2025, Sun et al. 2024a].
- **Semantic teacher branch (frozen at AE-train time):** a frozen VFM $f_{vfm}$ (DINOv2-L/14) followed by a learned **Downsampler** $E_{sp}$ that compresses 1024-dim VFM tokens into the $d$-dim latent space.

**Latent Normalization.** Numerical stability and a well-distributed latent are obtained with a *parameter-free* RMSNorm that projects the latent mean onto a unit hypersphere [Ke and Xue 2025 — hyperspherical latents], bounding values and preventing collapse:

$$\mu = \text{RMSNorm}(A_p(E_p(x)))$$

Here $\mu$ is the normalized latent **mean** (the deterministic code), living on $\mathbb{S}^{d-1}$ per token.

**Dynamic Noise Sampling** (σ-VAE [Sun et al. 2024a]). Rather than a fixed posterior variance, a noise scale $\sigma$ is drawn each step and added to $\mu$:

$$z = \mu + |\sigma|\odot\epsilon,\quad \epsilon\sim\mathcal{N}(0,1),\ \ \sigma\sim\mathcal{N}(0,C_\sigma)$$
$$\hat{x} = D_p(z)$$

where $C_\sigma$ is a hyperparameter setting the noise-sampling level (the variance of the per-step scale). Because $\sigma$ varies, the decoder must reconstruct across a *range* of perturbation magnitudes, learning a continuous manifold that tolerates the latent distribution shift a diffusion sampler will impose. ($C_\sigma$ is the key robustness knob — see Ablations.)

### Objective
Reconstruction uses the standard multi-term LDM loss:

$$\mathcal{L}_{pixel} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{lpips}\mathcal{L}_{lpips} + \lambda_{gan}\mathcal{L}_{gan} \tag{1}$$

with $\mathcal{L}_{rec}$ a pixel-level $L_1$ term, $\mathcal{L}_{lpips}$ a perceptual (LPIPS) term, and $\mathcal{L}_{gan}$ an adversarial term.

**Semantic Preservation loss** aligns the latent mean $\mu$ to the downsampled teacher features via MSE:

$$\mathcal{L}_{sp} = \|\mu - E_{sp}(f_{vfm}(x))\|_2^2 \tag{2}$$

This is **Latent Alignment**: supervision happens at the compressed bottleneck, where $E_{sp}(f_{vfm}(x))$ is the teacher target already reduced to the AE's $d$ dimensions. Total objective:

$$\mathcal{L}_{total} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{lpips}\mathcal{L}_{lpips} + \lambda_{gan}\mathcal{L}_{gan} + \lambda_{sp}\mathcal{L}_{sp} \tag{3}$$

with $\lambda_{rec}=1.0,\ \lambda_{lpips}=1.0,\ \lambda_{gan}=0.5,\ \lambda_{sp}=1.0$.

**Why remove KL?** Vanilla VAEs use a KL penalty to push the posterior toward $\mathcal{N}(0,I)$ so the space is samplable, but the weighted KL term is unstable and over-restrictive for diffusion. GAE substitutes a *hard geometric constraint* — RMSNorm onto the unit hypersphere keeps $\mu$ bounded and well-distributed, and dynamic noise sampling supplies the stochasticity that KL's variance term used to provide, without KL's instability. This preserves latent structure that is "more conducive to the subsequent denoising process."

### Latent Alignment — the design-space study (§4)

The central conceptual contribution. VFMs output high-dim features (e.g. 1024 for ViT-L) but diffusion wants compact latents (d=32). *Where* you reconcile this mismatch matters. Three paradigms (Fig. 3):

- **Pre Alignment:** align the encoder's high-dim intermediate features (before compression) to the VFM.
- **Post Alignment:** project the compressed latent *back up* to high-dim via an expansion layer, then supervise there.
- **Latent Alignment (theirs):** introduce a **semantic downsampler** that projects the VFM's high-dim output *down* to the latent's $d$ dims, and supervise the bottleneck directly.

**Pilot study** (ViT-L AE, frozen DINOv2-L/14 teacher, 60 epochs; for Latent Alignment a pre-computed **SVD** projection from 64,000 ImageNet samples maps 1024-dim → 32-dim). Result (Tab. 1, LP accuracy on stand-alone latents): Pre = 60.8, Post = 63.2, **Latent (SVD) = 60.9** LP with rFID comparable — and critically, Pre Alignment "suffers a dramatic loss in semantic discriminability," i.e. supervising before compression does not survive the bottleneck. The takeaway: **anchor at the bottleneck.** (Teacher/DINO upper bound LP ≈ 83.7.)

### Semantic Teacher training (§4.2)

A static SVD downsampler ignores spatial correlations, so they learn a **parametric** downsampler. Using the **Feature Autoencoder (FAE)** [Gao et al. 2025] recipe: frozen VFM $f_{vfm}$ + parametric downsampler $E_{sp}$ + a lightweight 4-layer LLaMA-style decoder $D_{sp}$, pre-trained with a **cosine distillation** objective on the VFM's patch tokens (treated as semantic anchors):

$$\mathcal{L}_{spt} = -\cos\big(D_{sp}(E_{sp}(f_{vfm}(x))),\ f_{vfm}(x)\big) \tag{4}$$

i.e. train the bottleneck so the discarded semantic content can be *directionally* recovered. After pre-training, $D_{sp}$ is thrown away and the frozen $E_{sp}$ becomes the fixed teacher target in Eq. (2). **Downsampler architecture matters** (Tab. 2): Single-Attn = 62.8 LP, Attn+Linear = 63.4, **Attn+Patch-Conv = 75.6** LP. The winning "Patch Convolution" (Algorithm 1) does window-partition → flatten → a single joint projection $W_{proj}\in\mathbb{R}^{(w^2\cdot C)\times(w^2\cdot d)}$ → reshape/window-reverse, a spatial-aware end-to-end map that dominates the token-wise baselines.

### Training procedure
- **Autoencoder:** ViT-L backbone, $f=16$, DINOv2-L frozen VFM. 200 epochs, global batch 1024, AdamW. Defaults $\lambda_{sp}=1.0$, $C_\sigma=0.2$, $d\in\{32,64\}$.
- **Diffusion generator:** LightningDiT-XL/1 [Yao et al. 2025a], constant lr $2.0\times10^{-4}$, batch 1024, EMA 0.9999. QK-Norm applied for the 800-epoch runs; omitted for 80-epoch benchmarks. RAE-style **time-shift** on the noise schedule.
- Ablations use 100-epoch AEs + 80-epoch LightningDiT-XL (no QK-Norm).

### Inference / sampling
250 sampling steps. **SDE sampler** [Song et al. 2020] for the no-CFG results; **ODE sampler** [Chen et al. 2018] when CFG is used. Class-uniform sampling; 50,000 images for both gFID and rFID. For d=32 @ 800 ep: timeshift 0.4, CFG interval 0.3, CFG weight 3.3; @ 80 ep: 0.4 / 0.25 / 2.5. For d=64: timeshift 0.5.

## Experimental setup
- **Dataset:** ImageNet-1K, class-conditional generation at 256×256.
- **Baselines:** autoregressive (MaskGIT, LlamaGen, VAR, MagViT-v2, MAR) and latent-diffusion (MaskDiT, DiT, SiT, FasterDiT, MDT/MDTv2, REPA, VA-VAE, REPA-E, AlignTok, **RAE**, **FAE**, VTP).
- **Metrics:** *generation* — gFID, Inception Score, Precision, Recall; *reconstruction* — rFID, PSNR, SSIM, LPIPS; *semantics* — Linear Probing (LP) accuracy on frozen latents.

## Key results
- **Headline (d=32, ImageNet 256², no CFG):** gFID **1.82 @ 80 ep**, **1.31 @ 800 ep**; with CFG **1.48 / 1.13**. This 80-epoch 1.82 already beats VA-VAE's *full* 800-epoch result — the efficiency claim.
- **vs. peers @ 800 ep no-CFG / CFG:** FAE 1.48 / 1.29; RAE(DiTDH-XL) 1.51 / 1.13* (*uses AutoGuidance + larger 839M model); REPA-E 1.70 / 1.15; AlignTok 2.04 / 1.37; VA-VAE 2.17 / 1.35. GAE (675M generator) matches RAE's 1.13 CFG using only *standard* CFG and a smaller model.
- **Semantic–reconstruction Pareto:** at d=32 GAE hits **69.4% LP vs VA-VAE's 43.1%** at the same dimension. At **d=64: 78.3% LP** (vs VTP-L 73.9%), rFID **0.382 vs FAE 0.660**, gFID 1.96 @ 80 ep (vs VTP-L 2.81), 1.53 @ 160 ep (vs FAE 2.02), 1.29 @ 800 ep.
- **Reconstruction (Tab. 10):** at d=64, PSNR 29.085 / SSIM 0.834, dramatically better than RAE (ViT-XL, d=768) at PSNR 18.86 / SSIM 0.62 — GAE reconstructs far better at a fraction of the latent dimension.
- **Convergence (Tab. 9):** d=32 gFID 1.82→1.51→1.37→1.33→1.31 over 80→160→320→480→640 epochs (plateau by ~640).

## Ablations
- **Semantic weight $\lambda_{sp}$ (Tab. 6):** $\lambda_{sp}=0$ → LP collapses to 5.74, gFID 12.55; 0.5 → LP 63.5, gFID 2.35; **1.0 → LP 69.2, gFID 2.36** (best balance); 2.0 → LP 71.4 but reconstruction degrades and gFID worsens to 2.45. More semantics helps until it starves reconstruction.
- **$C_\sigma$ / dynamic noise (Tab. 4, Fig. 4):** small $C_\sigma$ gives better rFID early (d=32, $C_\sigma$=0.05 → rFID 0.37) but poor robustness as noise grows; $C_\sigma$=0.2 chosen for both d=32 and d=64 as the robust/quality balance (gFID 2.32 at d=32). Fig. 4 shows higher-$\sigma$-trained decoders tolerate latent noise injection far better — RAE is exceptionally robust (stable even at σ=0.5), while VTP degrades sharply, explaining its weak gFID.
- **Latent Norm vs KL (Tab. 7):** ViT-L, no sp-loss, 100 ep — KL (weight 0.1) → rFID 0.977 / gFID 16.72; **Latent Norm → rFID 0.764 / gFID 12.55.** The hypersphere constraint beats KL for diffusion learnability.
- **Backbone (Tab. 8):** ViT-L (69.2 LP, gFID 2.36) > ViT-B (62.5 LP, 2.43) > ViT-S (rFID 1.55, LP n/a). Capacity drives both reconstruction and semantic preservation; ViT-L's better semantics yield better gFID despite marginally worse rFID than ViT-B.
- **Teacher choice (Tab. 11):** DINOv2-L (discriminative, 69.2 AE-LP → gFID 2.36) crushes MAE-L (reconstructive, 29.8 AE-LP → gFID 4.20) despite MAE's better rFID — semantic discriminability of the teacher, not its reconstruction, drives generation.

## Limitations
- **Paper's own:** an explicit rFID↔gFID trade-off governed by $C_\sigma$ (min-rFID settings hurt generative robustness); $\lambda_{sp}$ is a delicate knob (2.0 already over-aligns). A residual LP gap remains vs the teacher (69.2 vs the 75.6 downsampler bound / 83.7 raw DINO).
- **[analyst's view]:** all results are ImageNet-1K 256² class-conditional — no text-to-image, higher resolution, or video. "Geometric" oversells: the geometry is a unit-hypersphere normalization + bottleneck alignment, not a learned Riemannian metric or curvature-aware latent. Much of the pipeline is assembled from prior parts (σ-VAE noise sampling, FAE teacher, hyperspherical latents, LightningDiT generator, RAE time-shift), so novelty is the *combination* and the alignment-placement analysis rather than a new primitive. The parametric teacher adds a separate pre-training stage.

## Why it matters [analyst's view]

GAE is a clean empirical answer to a question the RAE/FAE/VA-VAE line has been circling: **inject VFM semantics at the compressed bottleneck, not before or after it**, and pair that with a geometric (hypersphere) latent constraint instead of KL. The 80-epoch 1.82 gFID is the practically important number — it says a well-anchored latent roughly halves the diffusion training budget. It also reframes "semantic tokenizer" design as a three-way equilibrium (compactness × semantics × noise-robustness) with an explicit knob for each ($d$, $\lambda_{sp}$, $C_\sigma$), which is a more legible design surface than the heuristics it critiques. For the vault this sits squarely next to the representation-autoencoder thread (RAE) and the "adapt VFMs as diffusion latents" cluster; the KL→hypersphere substitution echoes the broader move away from Gaussian-prior VAEs toward geometrically constrained latents.

## Open questions / things to verify
- Does Latent Alignment's bottleneck-supervision advantage hold at higher resolution / text-conditioned generation, or is it ImageNet-LP-specific?
- The teacher pre-training (Eq. 4, cosine distillation) adds cost — how much of GAE's gain is the *learned* downsampler vs. simply choosing DINOv2 + bottleneck alignment? (Tab. 1 SVD baseline already matches Pre/Post — worth probing.)
- RAE remains more noise-robust (stable to σ=0.5) yet GAE reconstructs better at far lower d — is there a hybrid that gets both?
- Sensitivity to VFM choice beyond DINOv2 vs MAE (e.g. SigLIP2, CLIP) is untested here.

## Connections
- Builds on / extends: [[papers/zheng-2025-rae-dit]] (RAE — the primary baseline it matches on gFID and beats on reconstruction; borrows the time-shift schedule)
- Related latent-autoencoder work: [[papers/liu-2026-improving-rae-reconstruction]], [[papers/li-2026-semantic-autoencoder]]
- Contrasts with (VFM-as-tokenizer / semantic-VAE line): FAE, VA-VAE, REPA-E, AlignTok, VTP — _needs note_ (not yet in vault)
- Adjacent generative-latent-geometry notes: [[papers/malnick-2026-designing-ot-flows]], [[papers/cai-2026-mode-mean-seeking]]; Riemannian-latent framing (weakly related, different sense of "geometry"): [[papers/knight-2026-riemannian-low-rank]], [[blogs/nicolas-riemannian-manifolds]]
- Topic MOCs: [[topics/diffusion-models]], [[topics/generative-models]], [[topics/representation-learning]], [[topics/self-supervised-learning]]
- Author indices: Hangyu Liu, Jianyong Wang, Yutao Sun — _needs note_ (no vault index yet)

## Selected quotes

> "By analyzing various alignment paradigms, GAE constructs an optimized low-dimensional semantic supervision target from VFMs to provide guidance for the autoencoder." — Abstract

> "we enforce a hard geometric constraint that projects the latent mean µ onto a unit hypersphere... This design preserves the structural integrity of the latent space, which is more conducive to the subsequent denoising process." — §3.2, Removal of KL Objective

> "supervising the high-dimensional features prior to the projector does not guarantee that semantic integrity is maintained through the subsequent compression stage. In contrast, Latent Alignment demonstrates a remarkable capacity to inherit VFM priors, achieving the highest LP accuracy." — §4.1
