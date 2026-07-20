---
type: paper
title: "Improving Reconstruction of Representation Autoencoder"
authors: ["Siyu Liu", "Chujie Qin", "Hubery Yin", "Qixin Yan", "Zheng-Peng Duan", "Chen Li", "Jing Lyu", "Chun-Le Guo", "Chongyi Li"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2602.08620
rw_id: 01kx5vf7ne2df3k64qb6v9xxyz
topics: [representation-learning, diffusion-models, generative-models]
priority: medium
read_state: queued
relevance: ""
added: 2026-07-14
last_updated: 2026-07-14
---

# Improving Reconstruction of Representation Autoencoder

> _Note: this paper was saved to the vault twice via Readwise — primary rw-id `01kx5vf7ne2df3k64qb6v9xxyz`, duplicate rw-id `01kx5vebvzjphbhg4cp3x4xh15` (same arXiv 2602.08620). This is the single canonical note; the dup can be dropped in a future dedup pass._

## TL;DR

Representation autoencoders (RAEs) — which repurpose frozen Vision Foundation Models (VFMs) like DINOv3 as the encoder for a latent diffusion model — get semantically clean, diffusion-friendly latents but reconstruct images poorly, because VFM features drop low-level detail (color, texture). This paper proposes **LV-RAE** (Local-Variations Augmented RAE): rather than fine-tune the VFM (which destabilizes the semantic manifold), it **freezes the VFM features as a fixed "base manifold" and trains a shallow 6-layer encoder to predict the residual low-level information**, which is added back to the semantic features before decoding. This hits PSNR 32.32 / SSIM 0.941 on COCO with CKNNA 0.987 semantic alignment (near-lossless), state-of-the-art reconstruction at an extreme f16d768 latent. The authors then diagnose a second problem — high-dim decoders are hypersensitive to off-manifold latent perturbations (adding 0.1σ noise drops PSNR from 32.32 to 17.72) — and fix it with a two-part noise strategy: fine-tune the decoder on randomly-scaled Gaussian-noised latents ($\sigma\sim\mathcal{U}(0,0.2)$) plus a GAN loss, and inject fixed noise into generated latents at sampling time. The full system reaches gFID 2.42 / IS 223.8 (DiTDH-XL, 800 epochs, w/o guidance) and gFDD 58.2, beating VA-VAE and SVG on the DINO-based distance metric.

## Context & motivation

Latent diffusion models (LDMs, Rombach et al. 2022) run diffusion in a VAE latent for efficiency. A recent line of work replaces or aligns the VAE latent with **VFM semantic features** (DINOv2/v3, SigLIP), because those features are linearly separable and easy for diffusion transformers to learn, giving faster convergence and better generation (REPA, VA-VAE, and the direct-RAE approach of Zheng et al. 2025). The problem: VFM features are optimized for high-level understanding and **lack low-level information**, so decoding them back to pixels is lossy — a bottleneck the paper cites as now limiting further LDM scaling (Labs et al. 2025 / FLUX, Esser et al. 2024 / SD3).

The paper's framing rests on a **manifold decomposition hypothesis** (Fig. 1): the real data manifold splits into (i) a smooth **base manifold** of global semantics — what VFMs capture — and (ii) **local variations** carrying low-level detail — what VFMs ignore. This directly motivates the design: don't relearn the base, just learn the missing local variations on top of it.

The naive fix — fine-tune the VFM with a reconstruction loss + an alignment loss — is shown to be sub-optimal: the two losses compete, the base manifold **drifts continuously** during training, and the local variations have no stable reference to attach to, so optimization is unstable and reconstruction/alignment both suffer (Table 2, Fig. 7).

Related work situated: prior RAE / VFM-tokenizer papers include RAE (Zheng et al. 2025), SVG (Shi et al. 2025, "latent diffusion without a VAE"), and several distill-VFM-into-low-dim approaches (Chen et al. 2025a, Bi et al. 2025, Gao et al. 2025). Diffusability regularization (Skorokhodov et al. 2025, EQ-VAE / Kouzelis et al. 2025) tackles high-frequency latent components. This paper is the RAE branch: keep the VFM whole, fix its reconstruction weakness.

## Method

### Problem formulation

Given an image $X\in\mathbb{R}^{H\times W\times 3}$, a frozen VFM $\Phi$ (default DINOv3) extracts patch-level semantic features $u\in\mathbb{R}^{N\times D}$ (taken **before** $\Phi$'s final LayerNorm), with $N=\frac{H}{16}\times\frac{W}{16}$ tokens. The goal is a latent $z\in\mathbb{R}^{N\times D}$ that (a) is decodable to a high-fidelity image and (b) stays maximally aligned with the semantic distribution $u$ so the downstream diffusion model still enjoys the "easy to learn" property. The latent config is **f16d768** (patch-16, 768 channels) — enormously higher-dimensional than a standard VAE (e.g. SD-VAE f8d4, FLUX f8d16).

### Core idea

Treat $u$ as a **fixed base manifold** and learn only the **residual** low-level information $r$ that is missing from it; the latent is $z = \text{LayerNorm}(r+u)$. Because $u$ never moves, the residual has a stable target and optimization converges cleanly.

### Architecture / algorithm

Both encoder $E$ and decoder $D$ are Transformers with RoPE positional embeddings (Su et al. 2024), patch size $16\times16$, T5-style MLP. Encoder is **lightweight (6 layers)** — it only has to learn low-level residuals; decoder is **deeper (12 layers)** to reconstruct pixels. Both hidden dim 768, 12 heads.

**Encoder input.** The image is projected to a patch representation $x_{\text{in}}\in\mathbb{R}^{N\times D}$ by a $16\times16$ convolution, then concatenated along the token axis with the semantic features:

$$x = [x_{\text{in}}; u] \in \mathbb{R}^{2N\times D}. \tag{1}$$

**Residual + latent.** Take the first $N$ tokens of the encoder output as the learned low-level residual, $r = E(x)[{:}N,:]\in\mathbb{R}^{N\times D}$, and form the latent by element-wise addition then LayerNorm:

$$z = \text{LayerNorm}(r+u). \tag{2}$$

Two initialization tricks keep semantics intact at step 0: the encoder's **final linear layer is zero-initialized** (so $r\approx 0$ initially and $z\approx\text{LayerNorm}(u)$), and the final LayerNorm is initialized with **the VFM's own final-LayerNorm parameters**. This means training starts exactly at "pure VFM features" and only adds detail from there.

**Reconstruction loss** — pixel $L_1$ plus LPIPS perceptual loss (Johnson et al. 2016; Zhang et al. 2018):

$$\mathcal{L}_{rec} = \alpha\,\mathcal{L}_1(X,\bar X) + \beta\,\mathcal{L}_{\text{Lpips}}(X,\bar X),\qquad \alpha=\beta=1. \tag{3}$$

**Alignment loss** — an explicit $L_2$ pull keeping the latent on the semantic manifold:

$$\mathcal{L}_{\text{align}} = \lVert z-u\rVert_2^2. \tag{4}$$

**Stage-I objective:**

$$\mathcal{L} = \mathcal{L}_{rec} + \eta\cdot\mathcal{L}_{\text{align}},\qquad \eta=5. \tag{5}$$

The result: reconstructions with PSNR ~32.32 while CKNNA (semantic alignment to $u$) stays ~0.99, and t-SNE / PCA (Figs. 3, 9) show LV-RAE latents overlapping DINOv3 features in a shared space.

### Derivations / why it works — the off-manifold sensitivity analysis

The high-dim latent decodes beautifully on clean $z$ but is **fragile**: the decoder amplifies perturbations orthogonal to the data manifold. The paper argues this via a **toy experiment** (Section 3.2). Sample 2-D points $\hat x\in\mathbb{R}^2$, embed into $D$ dims through a random orthonormal $P\in\mathbb{R}^{D\times2}$ ($P^\top P=I_2$), giving latents $z=P\hat x$. Train a small diffusion model (5-layer ReLU MLP, 512 hidden) in the $D$-dim latent. Construct a decoder that deliberately responds to **both** on- and off-manifold directions:

$$D(z) = P^\top z + \alpha\,\sin(\beta\,U^\top z)\,W, \tag{6}$$

where $U\in\mathbb{R}^{D\times(D-2)}$ is an orthonormal basis with columns orthogonal to $P$ ($P^\top U = 0$, i.e. the off-manifold subspace), and $W\in\mathbb{R}^{(D-2)\times2}$ has entries $\sim\mathcal{N}(0,\tfrac{1}{D-2}I)$. The Jacobian is:

$$J_D(z) = \frac{\partial D(z)}{\partial z} = P^\top + \alpha\beta\, W^\top \text{Diag}(\cos(\beta U^\top z))\, U^\top. \tag{7}$$

The **second term** is the entire off-manifold response, scaled by the total gain $\alpha\beta$. Setting $\beta=1$ leaves $\alpha$ as the single knob for off-manifold sensitivity. Findings (Fig. 4):
- $\alpha=0$: decoding is pure projection onto manifold directions; **increasing $D$ barely changes** the generated distribution — dimensionality is harmless.
- $\alpha>0$: as $D$ grows, generated samples deviate more and more from ground truth, up to severe structural collapse at large $\alpha$.

Intuition: a low-dim data manifold embedded in high-dim space has **many** orthogonal directions; the diffusion model can't hit the manifold exactly, and small off-manifold sampling errors get accumulated and amplified by a decoder with large off-manifold Jacobian gain. So **controlling off-manifold gain is essential for stable generation in high-dim latents** — exactly the regime RAEs live in. This is confirmed on real LV-RAE (Table 4): clean $z$ → PSNR 32.32, but $z+0.1\epsilon$ → 17.72, $z+0.2\epsilon$ → 13.68.

### Noise augmentation (Section 3.3) — the robustness fix

From the Jacobian view, **injecting noise during decoder training implicitly penalizes local gain** (it's a Tikhonov-style smoothing of the decoder response along all directions). Because different diffusion models leave different amounts of off-manifold error, the authors decouple robustness into a **tunable** parameter rather than baking in one fixed level.

**Stage-II training.** Freeze the LV-RAE encoder; fine-tune the decoder on a noise-perturbed latent

$$\tilde z = z + \sigma\cdot\epsilon,\quad \sigma\sim\mathcal{U}(0,\tau),\ \epsilon\sim\mathcal{N}(0,\mathbf{I}),\quad \tau=0.2, \tag{8}$$

with $\bar X = D(\tilde z)$. Crucially $\sigma$ is **randomly sampled per step** — a fixed $\sigma=0.1$ hurts clean reconstruction badly (implicit high-frequency truncation, decoder stops attending to fine latent detail), while random $\sigma$ preserves clean quality and adds robustness (Table 4). Add an adversarial (GAN) loss with **gradient-based adaptive weighting** at the decoder's final layer:

$$w_{\text{gan}} = \frac{\lVert\nabla\mathcal{L}_{rec}\rVert}{\lVert\nabla\mathcal{L}_{gan}\rVert}, \tag{9}$$

$$\mathcal{L} = \mathcal{L}_{rec} + \kappa\cdot w_{\text{gan}}\cdot\mathcal{L}_{gan},\qquad \kappa=0.75. \tag{10}$$

Discriminator is DINO-S/8 (following RAE), with differentiable augmentation + random crop to $224\times224$.

### Inference / sampling

At generation, given the diffusion model's output latent $z_0$, inject a **fixed** noise level:

$$\tilde z_0 = z_0 + \bar\sigma\cdot\epsilon,\quad \epsilon\sim\mathcal{N}(0,\mathbf{I}), \tag{11}$$

then decode $X = D(\tilde z_0)$. Larger $\bar\sigma$ → smoother output; both FID and FDD drop as $\bar\sigma$ rises, converging around $\bar\sigma\approx0.08$ (Fig. 8). This dynamically modulates the decoder's effective gain to absorb the diffusion model's off-manifold error. Diffusion backbones are LightningDiT-style **DiT-XL** and **DiTDH-XL** (DiT-XL + a DDT head), built on the RAE config; Euler ODE sampling, 250 steps, AutoGuidance (guiding model DiTDH-S trained 16 epochs, scale 1.4). RAE's **time-shift** schedule is used: $t' = \frac{\alpha t}{1+(\alpha-1)t}$, $\alpha=\sqrt{c\cdot h\cdot w/4096}\approx6.93$ for $h=w=16, c=768$.

### Training procedure

- **Data:** ImageNet-1K 256×256 (all training). VFM default DINOv3-B.
- **Autoencoder Stage I:** AdamW ($\beta_1=0.9,\beta_2=0.999$, wd 0.01), constant lr $1\times10^{-4}$, batch 512, **80k steps**.
- **Autoencoder Stage II (decoder noise/GAN):** encoder frozen; 10k steps rec-only, then +GAN for 90k steps.
- **Diffusion (DiT/DiTDH):** A100 40GB, bf16 mixed precision, AdamW lr $2\times10^{-4}$ for 40 epochs then decayed to $2\times10^{-5}$ through epoch 800, batch 1024, EMA decay 0.9995. SwiGLU, RMSNorm, Gaussian-Fourier timestep embedding, absolute PE + RoPE, QK-Norm.

## Experimental setup

- **Reconstruction eval:** ImageNet-1K val + COCO2017 val at 256×256; metrics PSNR, SSIM, LPIPS, and rFDD (Fréchet distance on DINOv2 features).
- **Semantic alignment:** CKNNA (Centered Kernel Nearest-Neighbor Alignment, Huh et al. 2024 / Platonic-rep) on first 5k ImageNet-val images.
- **Generation eval:** gFID, IS, Precision, Recall over 50k samples (10k for ablations); plus gFDD (DINOv2-feature Fréchet, argued more perception-correlated per Stein et al. 2023).
- **Baselines — reconstruction:** SD-VAE (f8d4), FLUX-VAE (f8d16), VA-VAE (f16d32), SVG (f16d392). **Generation:** ADM, RIN, PixelFlow, MaskDiT, DiT, SiT, FasterDiT, REPA, VA-VAE, DDT, SVG.

## Key results

**Reconstruction (Table 1, COCO / ImageNet).** LV-RAE f16d768 reaches **PSNR 32.32, SSIM 0.941, LPIPS 0.015** on COCO — best PSNR/SSIM of all tokenizers, beating FLUX-VAE (30.89) and crushing the other RAE-style SVG (21.71). With Stage-II noise augmentation, clean-latent reconstruction drops to PSNR 30.91 / SSIM 0.930 but **rFDD improves to 8.74** (from 13.20) — the robustness trade buys distributional fidelity.

**LV-RAE vs. fine-tuning the VFM (Table 2).** Across DINOv3-B / DINOv2-B / SigLIP2, LV-RAE beats F.T.-VFM on both reconstruction and alignment. DINOv3-B: LV-RAE PSNR 32.32 / SSIM 0.941 / **CKNNA 0.987** vs. F.T.-VFM 29.94–30.88 / CKNNA 0.925. DINOv3 is chosen as default because its features retain the most low-level info. SigLIP2 gains most in alignment (0.802 → 0.989).

**Generation (Table 3, ImageNet 256, w/o guidance).** DiT-XL + LV-RAE + 0.1 noise: gFID **3.77**, IS 185.4 at only 400 epochs. Scaling to DiTDH-XL (800 ep): gFID **2.42**, IS **223.8** (w/ guidance: gFID 1.82). Competitive with the strongest same-scale LDMs.

**FDD (Table 5).** At 800 epochs / 250 steps LV-RAE hits **gFDD 58.2**, well below VA-VAE (74.3) and SVG (130.4) — the headline win on the perception-correlated metric.

## Ablations

- **Fixed vs. drifting semantic space (Fig. 7):** LV-RAE keeps a lower, more stable alignment loss and converges to lower LPIPS than F.T.-VFM — evidence the "frozen base manifold" is what buys stability.
- **Decoder init:** the zero-init + VFM-LayerNorm-init slightly helps reconstruction, negligible effect on alignment/stability.
- **Noise schedule (Table 4):** fixed $\sigma=0.1$ tanks clean PSNR to 25.45 (high-freq truncation); random $\sigma\sim\mathcal{U}(0,0.2)$ keeps clean PSNR 30.91 while raising $z+0.2\epsilon$ robustness to 21.61 vs. 13.68 for the original decoder. → random-$\sigma$ adopted as default.
- **Inference noise (Fig. 8):** FID & FDD both fall with $\bar\sigma$, converging ~0.08; longer training (320→640 ep) mostly shifts curves vertically; architecture matters more than epochs (DiTDH-XL >> DiT-XL at low noise).
- **Encoder residual (Figs. 10–12):** dropping the encoder output $r$ at decode time yields color shifts and missing textures — confirms $r$ carries exactly the complementary low-level detail.

## Limitations

- Only ImageNet-256 class-conditional; **no text-to-image, no higher resolution** — the "reconstruction is the scaling bottleneck" motivation is not itself demonstrated at scale. _[analyst's view]_
- The reconstruction↔robustness trade is real: Stage-II noise augmentation **degrades clean reconstruction** (32.32 → 30.91). The best-reconstructing model and the best-generating model are not the same checkpoint. _[analyst's view]_
- Inference noise $\bar\sigma$ is an extra hand-tuned knob per diffusion backbone; the paper frames this as a feature (decoupling) but it is also a per-model calibration cost. _[analyst's view]_
- The off-manifold theory is argued through a **synthetic toy decoder** (Eq. 6) hand-built to have off-manifold response; it's illustrative, not a proof about the trained LV-RAE decoder's actual Jacobian spectrum. _[analyst's view]_
- Extreme f16d768 latent = 768 channels; storage/compute cost of diffusing such a wide latent is not discussed. _[analyst's view]_

## Why it matters [analyst's view]

This is a clean, well-motivated patch to the biggest weakness of the RAE paradigm (Zheng et al. 2025's "diffuse directly on frozen DINO features"): RAEs reconstruct badly, and this paper shows you can keep essentially all of their semantic-alignment / fast-convergence benefit while recovering VAE-grade reconstruction — **by not touching the VFM at all** and only predicting a residual. The "fixed base manifold + learned local variations" decomposition is a genuinely reusable framing for any "foundation-model-features-as-latent" pipeline, and the zero-init/LayerNorm-transplant trick that makes training start exactly at the pretrained features is the kind of detail worth stealing.

The second half — that **high-dimensional latents are intrinsically fragile at decode time because off-manifold directions get amplified** — is the more transferable insight. It reframes a lot of "diffusability" folklore (why wide semantic latents are hard to generate cleanly) as a decoder-Jacobian problem, and the fix (train-time random noise + inference-time noise injection to modulate effective gain) is simple and general. It connects to the broader intuition that adding noise = smoothing = penalizing local gain, familiar from denoising/consistency work.

Sits directly adjacent to the RAE-DiT line and the "VFM as tokenizer" cluster; contrasts with SVG (also VAE-free but reconstructs poorly here) and with distill-into-low-dim approaches by going the opposite way — keep the latent huge, fix the decoder instead.

## Open questions / things to verify

- Does the off-manifold amplification story hold if you measure the **real** LV-RAE decoder's Jacobian singular values along estimated off-manifold directions, not just the toy decoder?
- Is inference-time noise injection equivalent to (or dominated by) simply training the diffusion model better / to lower off-manifold error? The Fig. 8 "longer training just shifts curves" result hints the noise is compensating for a fixable model deficiency.
- How does this interact with **guidance** — AutoGuidance already perturbs the sampling trajectory; is inference noise partly redundant with it?
- Does the approach transfer to text-to-image and to video latents, where the manifold and the base-vs-detail split are messier?
- CKNNA 0.987 "near-lossless semantic alignment" — does that actually preserve the downstream linear-probe / REPA-style benefits, or only nearest-neighbor structure?

## Connections

- Builds on: [[papers/zheng-2025-rae-dit]] — the direct-RAE (diffuse on frozen DINO features) framework this paper fixes; LV-RAE inherits its DiT/DiTDH config, time-shift schedule, AutoGuidance, and DINO-S/8 discriminator.
- Sibling / same cluster: [[papers/liu-2026-geometric-autoencoder]], [[papers/li-2026-semantic-autoencoder]] — concurrent work on VFM-features-as-latent autoencoders.
- Contrasts with: SVG (Shi et al. 2025, "LDM without a VAE") `_needs note_`; VA-VAE / LightningDiT (Yao et al. 2025) `_needs note_`; REPA (Yu et al. 2025) `_needs note_`.
- Related in vault: [[papers/pao-huang-2026-flux-matching]] — flow-matching generative modeling; [[papers/huang-2026-semantic-tube-prediction]] — semantic-feature prediction.
- Background methods cited: FLUX-VAE (Labs et al. 2025) `_needs note_`, SD3 rectified-flow (Esser et al. 2024) `_needs note_`, DINOv3 (Siméoni et al. 2025) `_needs note_`, Platonic Representation Hypothesis / CKNNA (Huh et al. 2024) `_needs note_`.
- Topic MOCs: [[topics/representation-learning]], [[topics/diffusion-models]], [[topics/generative-models]]
- Author indices: _no existing author index for any LV-RAE author (Siyu Liu, Chongyi Li, Chun-Le Guo, …) — `_needs note_` if one is created._

## Selected quotes

> "We hypothesize that the real data manifold can be decomposed into two distinct components: a smooth base manifold representing global semantics (captured by VFMs) and local variations representing low-level information (ignored by VFMs)." — Fig. 1 caption

> "Instead of forcing the encoder to extract both semantic and low-level information into a shared latent, we treat the VFM semantic features as a fixed base manifold. The encoder is then tasked solely with learning the low-level information that is missing from the semantic features." — §3.1

> "Higher-dimensional latent spaces provide more off-manifold directions, along which even small deviations can accumulate and be amplified by the decoder." — §3.2

> "From a Jacobian perspective, noise injection implicitly penalizes excessive local gain, reducing the decoder's sensitivity along all directions." — §3.3
