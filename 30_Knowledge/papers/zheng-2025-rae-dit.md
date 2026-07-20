---
type: paper
title: "Diffusion Transformers with Representation Autoencoders"
authors: ["Boyang Zheng", "Nanye Ma", "Shengbang Tong", "Saining Xie"]
year: 2025
venue: "arXiv 2025"
url: https://arxiv.org/abs/2510.11690
rw_id: 01kx6cmjvqyccx1r6e8z3h284t
topics: [diffusion-models, generative-models, representation-learning, self-supervised-learning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-14
last_updated: 2026-07-16
---

# Diffusion Transformers with Representation Autoencoders

## TL;DR

The paper replaces the workhorse SD-VAE of latent diffusion with a **Representation Autoencoder (RAE)**: a *frozen* pretrained representation encoder (DINOv2, SigLIP2, or MAE) paired with a *trained* lightweight ViT decoder, and trains Diffusion Transformers directly in that high-dimensional, semantically rich latent space. This overturns two received beliefs — that semantic encoders can't reconstruct pixels (RAE with MAE-B hits rFID 0.16 vs SD-VAE's 0.62) and that diffusion fails in high-dimensional latents. The key enabling insight is that **the DiT width must be at least as large as the encoder's token dimension** (proved via a rank argument, Theorem 1), plus a dimension-dependent noise-schedule shift and noise-augmented decoder training. A shallow-but-wide "DDT head" (yielding the DiTDH variant) lets width scale without quadratic FLOPs. On ImageNet the RAE-based DiTDH-XL reaches **1.51 FID at 256×256 without guidance and 1.13 with AutoGuidance** (also 1.13 at 512×512), with a reported ~47× training speedup over SiT-XL and no auxiliary REPA-style alignment loss. The authors argue RAE should be the new default for DiT training.

## Context & motivation

Latent diffusion (LDM, DiT/SiT) has become the dominant image/video generation recipe, but while the diffusion backbone evolved rapidly, the autoencoder defining the latent space "has barely evolved." The standard SD-VAE has three problems the paper attacks: (1) an outdated convolutional backbone that is computationally heavy (≈6× encoder / 3× decoder GFLOPs vs RAE); (2) a heavily channel-compressed, low-dimensional latent (256² image → 32²×4 latent) that caps information capacity; and (3) a reconstruction-only training objective that yields latents with local appearance but weak global semantic structure — SD-VAE latents linear-probe at only ~8% top-1 on ImageNet.

Meanwhile self-supervised / multimodal encoders (DINO, MAE, JEPA, CLIP/SigLIP) learn semantically structured features, but latent diffusion stayed isolated from them. Prior attempts to inject semantics — REPA-style alignment (REPA, VA-VAE, REPA-E), DDT, REG, ReDi — do so *indirectly* via extra training stages and auxiliary losses. Two "long-standing assumptions" blocked the direct route: semantic encoders are believed unsuitable for faithful reconstruction, and diffusion is believed to perform poorly in high-dimensional latents. The paper's contribution is to show **both assumptions are wrong**, and that diffusing directly in a frozen representation space is not only feasible but faster-converging and higher-quality — with no auxiliary alignment loss.

## Method

### Problem formulation

Given an image $x \in \mathbb{R}^{3\times H\times W}$ and a frozen representation encoder $E$ with patch size $p_e$ and hidden size $d$, encoding gives $N = HW/p_e^2$ tokens each of channel dimension $d$: $z = E(x)$. A ViT decoder $D$ with patch size $p_d$ maps tokens back to pixels; with the default $p_d = p_e$ the reconstruction $\hat{x} = D(z)$ matches input resolution. No channel compression is applied — the latent lives in the encoder's full $d$-dimensional token space. For 256×256 images the encoder produces 256 tokens, matching the sequence length of SD-VAE-based DiTs, so downstream diffusion incurs essentially no extra compute (channels are projected to the DiT hidden width in the first layer; patchification is <1% of GFLOPs).

The generative model is trained with a **flow-matching / linear-interpolant** objective: $x_t = (1-t)x + t\varepsilon$ with $x \sim p(x)$, $\varepsilon \sim \mathcal{N}(0,I)$, $t \in [0,1]$, and the network predicts the velocity $v(x_t,t)$ targeting $\varepsilon - x$.

### Core idea

Freeze a semantic encoder, train only a decoder on top of it to form the autoencoder, and run diffusion directly in that high-dimensional semantic latent — "linking semantic and generative modeling through a shared latent representation." Higher latent dimensionality, contrary to prior belief, becomes an *advantage* (richer structure, faster convergence) rather than a liability, at no extra token-count cost.

### Architecture / algorithm (the deepest part)

**RAE decoder training.** The decoder is trained with the standard VAE-style reconstruction mixture of L1, LPIPS and adversarial (GAN) losses:

$$z = E(x), \quad \hat{x} = D(z)$$
$$\mathcal{L}_{rec}(x) = \omega_L\, \text{LPIPS}(\hat{x},x) + \text{L1}(\hat{x},x) + \omega_G\, \lambda\, \text{GAN}(\hat{x},x)$$

with $\omega_L = 1.0$, $\omega_G = 0.75$. $\lambda$ is the adaptive GAN weight from Esser et al. (2021), balancing reconstruction vs adversarial gradient scales:

$$\lambda = \frac{\|\nabla_{\hat{x}}\mathcal{L}_{rec}\|}{\|\nabla_{\hat{x}}\text{GAN}(\hat{x},x)\| + \epsilon}$$

The discriminator is a frozen DINO-S/8 backbone (StyleGAN-T-style), with the GAN loss from Esser et al. (2021). Encoders studied: **DINOv2-B** ($p_e=14$, $d=768$), **SigLIP2-B** ($p_e=16$, $d=768$), **MAE-B** ($p_e=16$, $d=768$); DINOv2 also at S/B/L ($d=384/768/1024$). Default decoder is ViT-XL. DINOv2 is the default encoder for generation (best gFID empirically, even though MAE has the best rFID).

**Three fixes that make DiT work on RAE latents.** Out of the box the standard recipe fails: DiT-S completely collapses (gFID 215.76 on RAE vs 51.74 on SD-VAE) and even DiT-XL underperforms (23.08 vs 7.13). The paper diagnoses and fixes three causes:

1. **Width ≥ token dimension (Section 4.1).** In a single-image overfitting probe, samples are poor when DiT width $d < $ token dimension $n = 768$ and improve sharply once $d \geq n$; losses converge only when $d \geq n$. Doubling *depth* (12→24) at fixed width $d=384$ does *not* fix it, ruling out generic capacity. So width must match/exceed the encoder token dimension. For DINOv2-B ($n=768$) they use DiT-XL. This motivates Theorem 1 below.

2. **Dimension-dependent noise-schedule shift (Section 4.2).** Prior resolution-based schedule shifts were derived for few-channel (C≤16) VAE/pixel inputs and ignore channel dimensionality. Since Gaussian noise is added over both spatial and channel dims, more channels means the *effective* resolution per token grows, corrupting information less at a given noise level. They generalize the shift to the **effective data dimension** = (#tokens × per-token dimensionality). Using the Esser et al. (2024) shift, for a base schedule $t_n \in [0,1]$ and dims $n, m$, the shifted timestep is

   $$t_m = \frac{\alpha\, t_n}{1 + (\alpha - 1)\, t_n}, \qquad \alpha = \sqrt{m/n}$$

   with base $n = 4096$ and $m$ set to the RAE effective data dimension. This alone cuts gFID from 23.08 → 4.81 (Table 4).

3. **Noise-augmented decoding (Section 4.3).** VAE decoders are trained on a continuous latent distribution $\mathcal{N}(\mu,\sigma^2 I)$ and tolerate small noise; the RAE decoder is trained only on the *discrete* empirical latent set $p(z) = \sum_i \delta(z - z_i)$ and sees an OOD shift when the diffusion model emits slightly noisy/continuous latents at inference. Fix: train the decoder on a smoothed distribution by adding $n \sim \mathcal{N}(0,\sigma^2 I)$,

   $$p_n(z) = \int p(z - n)\,\mathcal{N}(0,\sigma^2 I)(n)\, dn$$

   with $\sigma$ itself sampled stochastically from $|\mathcal{N}(0,\tau^2)|$ for regularization. This improves gFID (4.81 → 4.28) at a slight rFID cost (0.49 → 0.57) — an expected trade-off since smoothing removes fine detail.

**DiTDH — the wide DDT head (Section 5).** Scaling DiT *width* across the whole backbone to meet the token-dimension requirement is quadratically expensive. Inspired by DDT (Wang et al., 2025c) but from a different motivation, they attach a **shallow-but-wide** transformer head $H$ dedicated to denoising on top of a base DiT $M$. Given noisy input $x_t$, timestep $t$, optional label $y$:

$$z_t = M(x_t \mid t, y), \qquad v_t = H(x_t \mid z_t, t)$$

The head is 2 layers, 2048-dim for all DiTDH sizes. This raises effective width without quadratic FLOP growth: DiTDH-B beats DiT-XL using ~40% of the training FLOPs; DiTDH-XL reaches FID 2.16 at 80 epochs, nearly half DiT-XL's 4.28.

### Derivations / why it works

**Theorem 1 (width lower bound).** For $x \sim p(x) \in \mathbb{R}^n$, $\varepsilon \sim \mathcal{N}(0,I_n)$, $x_t = (1-t)x + t\varepsilon$, consider the "bottlenecked" function family

$$\mathcal{G}_d = \{\, g(x_t,t) = B f(A x_t, t) : A \in \mathbb{R}^{d\times n},\ B \in \mathbb{R}^{n\times d},\ f:[0,1]\times\mathbb{R}^d \to \mathbb{R}^d \,\}$$

where $d < n$ ($A$, $B$ are the DiT input/output projections and $f$ the stack of DiT blocks whose width $d$ is below the token dimension $n$). Then the flow-matching loss obeys

$$\mathcal{L}(g,\theta) = \int_0^1 \mathbb{E}_{x,\varepsilon}\big[\, \| g(x_t,t) - (\varepsilon - x) \|^2 \,\big]\, dt \ \geq\ \sum_{i=d+1}^{n} \lambda_i$$

where $\lambda_i$ are the eigenvalues of the covariance of the target $W = \varepsilon - x$. When $d \geq n$, $\mathcal{G}_d$ contains the unique minimizer. **Intuition:** a width-$d$ transformer maps its input through a rank-$d$ bottleneck ($A$ then $B$), so it can only represent the velocity field within a $d$-dimensional subspace; the residual energy in the top $n-d$ eigen-directions of the target is irreducible. In the toy case $p(x) = \delta(x - x_0)$, $W \sim \mathcal{N}(-x, I_n)$ so all $\lambda_i = 1$ and the normalized bound becomes $\tilde{\mathcal{L}} \geq (n-d)/n$, which matches the empirical loss curves in Figure 3. A multi-width/multi-encoder overfitting grid (Table 3) confirms convergence only when model width ≥ token dim (DiT-B converges on DINOv2-B; DiT-S does not).

The paper also reframes the apparent conflict with the low-intrinsic-dimension manifold hypothesis: injecting Gaussian noise into $x_t$ throughout training "diffuses" the data manifold into a full-rank one, so required capacity scales with the *ambient* data dimension, not the manifold's.

### Training procedure

- **Decoder (Table 12):** Adam, betas (0.5, 0.9), max lr 2e-4 → min 2e-5 cosine decay, weight decay 0, batch size 512, 1-epoch warmup, 16 training epochs (discriminator 10 epochs, DINO-S/8 frozen). LPIPS from epoch 0, discriminator from epoch 6, adversarial loss from epoch 8. Augmentation: resize to 384×384 then random-crop to 256×256, plus differentiable augmentations before the discriminator.
- **DiT diffusion:** follows LightningDiT (Yao et al., 2025) — AdamW, constant lr 2.0e-4, batch size 1024, EMA 0.9999. Patch size 1 on RAE (seq length 256). Default 80 epochs on ImageNet-256; best results at 720/800 epochs.
- **DiTDH:** LightningDiT recipe caused loss spikes late and slow EMA convergence early, so instead: linear lr decay 2.0e-4 → 2.0e-5 with a 40-epoch constant warmup, EMA changed to 0.9995, gradient clipping 1.0.
- Trained with PyTorch/XLA on TPU (Google TRC); only EMA model performance reported.

### Inference / sampling

Standard ODE sampling with the **Euler sampler, 50 steps** (performance converges above 50 steps). gFID computed on 50K samples using class-balanced sampling (50 images/class). Best numbers use **AutoGuidance** (Karras et al., 2025). For 512×512, the decoder patch size can be set to $p_d = 2p_e$ so a diffusion model trained at 256 can be reused with a 2× upsampling decoder — no retraining — producing 512 outputs from the same 256 tokens.

## Experimental setup

- **Dataset:** ImageNet-1K, class-conditional generation at 256×256 and 512×512.
- **Metrics:** rFID (reconstruction FID on the ImageNet val set), gFID (FID-50K on generated samples), Inception Score, Precision, Recall, plus linear-probing top-1 for representation quality.
- **Baselines:** SD-VAE (reconstruction); for generation — autoregressive (VAR, MAR, xAR), pixel diffusion (ADM, RIN, PixelFlow, PixNerd, SiD2), VAE latent diffusion (DiT, SiT, MaskDiT, MDTv2, VA-VAE, REPA, DDT, REPA-E), and GAN/other at 512 (BigGAN-deep, StyleGAN-XL, MAGVIT-v2, EDM2).

## Key results

- **Reconstruction:** RAE beats SD-VAE (rFID 0.62) with every encoder — MAE-B 0.16, DINOv2-B / SigLIP2-B ≈ 0.49–0.53. rFID is stable across encoder sizes (DINOv2-S/B/L); larger decoders help (ViT-B 0.58 → ViT-XL 0.49). Even ViT-B decoder beats SD-VAE at 14× fewer GFLOPs.
- **Representation:** RAE inherits the frozen encoder's linear-probe accuracy (DINOv2-B ~84.5% top-1) vs SD-VAE's ~8%.
- **Generation (ImageNet 256, Table 8):** DiTDH-XL (DINOv2-B) — **1.51 gFID w/o guidance** and **1.13 w/ AutoGuidance** at 800 epochs; 2.16 gFID at just 80 epochs. Outperforms all prior methods (e.g. REPA-E 1.70/1.15, DDT, SiT, REPA) by a large margin. DiT-XL on RAE alone (no head) reaches 1.87/1.41.
- **Generation (ImageNet 512, Table 7):** DiTDH-XL 1.13 gFID with guidance at 400 epochs, beating EDM2 (1.25) and DDT (1.28).
- **Convergence:** RAE DiT-XL hits gFID 4.28 at 80 epochs and 2.39 at 720 — reported **47× training speedup over SiT-XL** and **16× over REPA-XL**; DiTDH-XL surpasses REPA/MDTv2/SiT-XL at ~5e10 GFLOPs and is best overall by ~5e11 GFLOPs (>40× less compute).

## Ablations

- **Schedule shift:** gFID 23.08 → 4.81 (Table 4).
- **Noise-augmented decoding $p_n(z)$:** gFID 4.81 → 4.28, rFID 0.49 → 0.57 (Table 5).
- **DiTDH vs DiT across encoder sizes (Table 6):** DiTDH-XL beats DiT-XL on DINOv2-S/B/L (e.g. DINOv2-L: 6.09 → 2.73); advantage grows with encoder size / latent dimensionality.
- **Does DiTDH need RAE? (Table 10):** On SD-VAE, DiTDH-XL (11.70) is *worse* than DiT-XL (7.13) — the wide head helps only in high-dimensional latents.
- **Is structure needed, or just high dimension? (Table 11):** On raw pixels (16×16×3=768 dim, matching DINOv2-B), DiTDH beats DiT (30.56 vs 51.09) but both are far worse than on RAE (2.16 / 4.28) — **high dimensionality alone is insufficient; structured representation is essential.**
- **Decoder upsampling for 512 (Table 9):** upsampling decoder gives gFID 1.61 (vs 1.13 direct) using 256 tokens instead of 1024 — 4× more efficient at a modest quality cost.
- **FID protocol:** class-balanced sampling is ~0.1 FID lower than uniform random; baselines re-evaluated for fairness, which raised some reported baseline numbers.

## Limitations

Paper-acknowledged: noise-augmented decoding trades a small rFID loss for gFID gain; decoder-upsampling to 512 raises rFID (0.53 → 0.97) and gFID (1.13 → 1.61) vs direct training. The DDT head provides no benefit (even hurts) in low-dimensional VAE latents, so it is not a general-purpose improvement. The authors also caution that absolute FID becomes less meaningful as fidelity saturates. [analyst's view] Additional honest flags: results are ImageNet-only (no text-to-image or video despite the framing); encoder choice for generation (DINOv2) is picked empirically and not fully explained (best rFID encoder MAE is *not* the best generator); the "no extra compute" claim holds for token count but the wider DiT and the extra 2-layer/2048-dim head do add parameters (676M → 839M) and FLOPs; and the width≥dim requirement means larger encoders force wider (costlier) backbones absent the head.

## Why it matters [analyst's view]

This is a clean, potentially recipe-changing result: it removes the VAE from latent diffusion and shows that a frozen self-supervised encoder plus a cheap decoder is strictly better on reconstruction, representation, *and* generation, while dropping the entire REPA-style auxiliary-alignment machinery. If it replicates beyond ImageNet, "the autoencoder is a representation foundation, not a compressor" becomes the default mental model, unifying the understanding stack (DINO/SigLIP) with the generation stack. The width-vs-token-dimension theorem is a genuinely useful piece of theory that explains a lot of prior high-dimensional-diffusion pain and gives a concrete design rule. It sits directly next to the wave of RAE-improvement work in the vault (LV-RAE reconstruction fixes, geometric autoencoders, delta-tokens) — this is the seed paper those extend.

## Open questions / things to verify

- Why does DINOv2 generate best while MAE reconstructs best? The decoupling of rFID from gFID is asserted, not explained.
- Does the approach transfer to text-to-image and video, where the "fixed token count" and encoder availability assumptions are weaker?
- The 47×/16× speedup claims depend on the re-evaluated (class-balanced) baselines and specific budgets — worth checking against original reported numbers.
- Theorem 1's bound is exact for the linear-bottleneck family $\mathcal{G}_d$; real DiTs have nonlinear attention mixing across tokens — how tight is the width rule in practice for very high-dim encoders (e.g. DINOv2-g)?
- Robustness of noise-augmented decoding: how sensitive is gFID to $\tau$ and to the encoder?

## Connections

- Extends to / extended by: [[papers/liu-2026-improving-rae-reconstruction]] (LV-RAE — improves RAE reconstruction fidelity)
- Related: [[papers/liu-2026-geometric-autoencoder]], [[papers/kerssies-2026-delta-tokens]]
- Contrasts with: REPA / VA-VAE / REPA-E / DDT style semantic-alignment methods (this paper diffuses *directly* in the representation space instead of aligning to it)
- Topic MOCs: [[topics/diffusion-models]], [[topics/generative-models]], [[topics/representation-learning]], [[topics/self-supervised-learning]]
- Author indices: [[authors/shengbang-tong]], [[authors/saining-xie]], [[authors/boyang-zheng]], [[authors/nanye-ma]]

## Selected quotes

> "frozen representation encoders, even those explicitly optimized for semantics over reconstruction, can be repurposed into powerful autoencoders for generation, yielding reconstructions superior to SD-VAE without architectural complexity or auxiliary losses." — §1

> "for generation in RAE's latent space to succeed, the diffusion model's width must match or exceed the RAE's token dimension." — §4.1

> "high dimensionality alone is not sufficient: the structured representation provided by RAE is crucial for achieving strong performance gains." — §6.3

> "RAE offers clear advantages and should be the new default for diffusion transformer training." — Abstract
