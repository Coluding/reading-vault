---
type: paper
title: "Beyond Language Modeling: An Exploration of Multimodal Pretraining"
authors: ["Shengbang Tong", "David Fan", "John Nguyen", "Ellis Brown", "Gaoyue Zhou", "Shengyi Qian", "Boyang Zheng", "Théophane Vallaeys", "Junlin Han", "Rob Fergus", "Naila Murray", "Marjan Ghazvininejad", "Mike Lewis", "Nicolas Ballas", "Amir Bar", "Michael Rabbat", "Jakob Verbeek", "Luke Zettlemoyer", "Koustuv Sinha", "Yann LeCun", "Saining Xie"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2603.03276
rw_id: 01ks5mjw1pv57jart7tvrccrb0
topics: [multimodal-pretraining, mixture-of-experts, scaling-laws, vision-language, world-models]
priority: high
read_state: skimmed
relevance: ""
added: 2026-05-21
last_updated: 2026-06-10
---

# Beyond Language Modeling: An Exploration of Multimodal Pretraining

_Note: re-saved to Readwise 2026-06-10 under a second rw-id `01ktsjtt3wshk99cmatef8hy2k`; original (frontmatter) rw-id is the durable one. FAIR/Meta + NYU. Project page: beyond-llms.github.io. Default model 2.3B params (1.5B active)._

## TL;DR

A large-scale, **from-scratch, controlled** study of native multimodal pretraining (text + image + video + action-conditioned video) built on the **Transfusion** recipe: one decoder-only transformer, **next-token prediction for language and flow-matching diffusion for vision**, jointly from random init (no LLM warm-start, so multimodal dynamics are isolated from inherited language knowledge). Four headline takeaways (the paper's "Suggestions"): **(S1)** a single **Representation Autoencoder (RAE)** — a frozen *semantic* encoder (SigLIP 2) decoded back to pixels — beats VAEs at *both* visual understanding and generation, collapsing the usual dual-encoder design; **(S2)** vision and language data are complementary — vision barely dents text perplexity and diverse pretraining beats task-specific data even at 5× less in-domain volume; **(S3)** **world-modeling emerges** from generic multimodal pretraining (navigation with actions encoded as plain text tokens, no architecture change, ~1% in-domain data needed); **(S4)** **Mixture-of-Experts** is the architectural fix for a **scaling asymmetry** the paper measures via IsoFLOP — vision is significantly more data-hungry than language, and sparse MoE lets language get the capacity it needs while vision soaks up data, narrowing the scaling-exponent gap from 0.10 to 0.05.

## Context & motivation

Foundation models have been defined by language pretraining, but the authors argue text is "a lossy compression of reality" (Plato's-cave framing) and high-quality text is "approaching exhaustion," while the visual world is an endless signal stream. The scientific problem: **the design space for native multimodal pretraining is opaque** because most prior unified models (BAGEL/Deng 2025, Liao 2025, etc.) **initialize from pretrained LLMs**, which confounds what is *learned from multimodal training* vs *inherited from language pretraining*. This paper's whole contribution is methodological — train everything from scratch under controlled compute and isolate the factors (visual representation, data, architecture, scaling) one at a time. It builds directly on **Transfusion** (Zhou et al. 2025a), and contrasts with the discrete-token-vision line (Chameleon-style) and the adapter-bridging line (LLM ↔ frozen diffusion model).

## Method

### Problem formulation — one model, two objectives

A single decoder-only transformer ingests an interleaved sequence of text tokens **T** and visual tokens **I**, and is trained with a **weighted sum of two losses**, one per modality.

**Language** — standard autoregressive cross-entropy:
$$\mathcal{L}_{LM} = -\sum_{i=1}^{n} \log p_\theta(x_i \mid x_{<i})$$

**Vision** — *image-wise flow matching* on continuous latents. Let $z_0$ be the clean latents of an image/frame (flattened to a token sequence), $\epsilon \sim \mathcal{N}(0,I)$, $t \sim U[0,1]$, and the interpolant $z_t = (1-t)\,\epsilon + t\,z_0$. The model predicts a velocity field and regresses to the straight-line target:
$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t,z_0,\epsilon}\big[\,\lVert v_\theta(z_t, t, \text{context}) - (z_0 - \epsilon) \rVert_2^2\,\big]$$

**Joint objective:**
$$\mathcal{L} = \lambda_{LM}\,\mathcal{L}_{LM} + \lambda_{\text{flow}}\,\mathcal{L}_{\text{flow}}, \qquad \lambda_{LM}=1.0,\ \lambda_{\text{flow}}=3.0 \text{ (default)}$$

Two non-obvious training details:
- **Noise schedule shifted toward the noisier end** (Esser et al. 2024), following RAE — necessary to diffuse through *high-dimensional* visual representations.
- **One independent $t$ per image/frame**, applied to all tokens in that image. This per-image independent noise aligns the dynamics with **Diffusion Forcing** (Chen et al. 2024a) — different frames in a sequence sit at different noise levels, which is what later enables autoregressive video rollout / world modeling.
- They also explore an alternate parameterization **x-pred** (predict $z_0$ directly) vs the default **v-pred** (velocity); which one wins turns out to depend on the visual representation (see Architecture).

### Architecture — the part to read closely

**Backbone.** Decoder-only transformer à la Transfusion, trained from scratch on the vision+language mixture. **Key simplification vs Transfusion: linear projection layers instead of a U-Net** for the visual pathway — justified because they do *not* resample the number of visual tokens for most encoders, so a direct linear map into the transformer's latent space suffices.

**Modality-specific FFNs (the default capacity-separation move).** Inside each self-attention block, **two separate FFNs — one for text tokens, one for vision tokens** — rather than a shared FFN. Each token only activates its modality's FFN, so this is *capacity separation at no extra inference cost*. This single change "unanimously improves performance" (Fig. 3): lower text PPL **and** better image generation **and** better VQA. The **default model is 2.3B total params, 1.5B activated per token** (only one FFN fires); a shared-FFN dense equivalent would be 1.5B total.

**Tokenization.**
- *Text*: LLaMA-3 BPE tokenizer, next-token prediction. Denoted **T**.
- *Vision*: each image / video frame → latent tokens via a **frozen visual encoder** (architecture is *encoder-agnostic*). **Videos are encoded frame-by-frame.** All frames pre-processed to **224×224**. Denoted **I**. Default encoder: **SigLIP 2 So400M**.

**Attention masking (hybrid, via FlexAttention).** Text uses a **standard causal mask**. Vision uses a **block-wise causal mask**: each frame (or static image) is a block; tokens within a frame attend **bidirectionally** to each other *and* causally to all previous tokens. This single scheme lets the model process arbitrary interleavings of any modality in sequence.

**Visual representation study (§3) — the RAE finding.** Three encoder families compared: (1) **VAEs** (SD-VAE, FLUX.1) — low-dimensional latents; (2) **semantic encoders** — language-supervised **SigLIP 2 So400m**, self-supervised **DINOv2-L** and **WebSSL-L**; (3) **raw pixels** (each 14×14 RGB patch flattened into a token, optionally with x-pred à la JiT). For the semantic encoders, latents are decoded back to pixels with the **off-the-shelf RAE decoder** — this *is* the "RAE" representation. Findings: all representations leave text PPL ~unchanged (raw pixels best on text); **semantic encoders beat VAEs on both understanding and generation**, SigLIP 2 best overall (e.g. beats FLUX.1 on VQA, DPGBench, GenEval). Raw pixels underperform on generation but are surprisingly close on VQA → flagged as a promising future direction. ⟹ **a single SigLIP-2-based RAE replaces the conventional dual-encoder (VAE-for-gen + SigLIP-for-understanding) stack.**

**Mixture-of-Experts (§6) — generalizing the FFN split.** Instead of hand-fixing two modality FFNs, let MoE *learn* the capacity decomposition by routing each token to a subset of experts. Three MoE axes studied (default MoE setup: 57B tokens, 50/50 DCLM-text/Shutterstock-image):
- **Granularity** $G = \frac{4\,d_{\text{model}}}{d_{\text{expert}}}$. Sweep $G \in \{1,4,16,32,64\}$ at **fixed active compute** (as $G$ grows, more experts activate but each is smaller). $G{=}1$ → 16 big experts (d=8192, Top-1); $G{=}64$ → 1024 tiny experts (d=128, Top-64). **Vision saturates at $G{=}4$, language at $G{=}16$** (language benefits more from fine-grained routing); plateau past $G{=}32$ ⟹ **adopt $G{=}16$**. All these models: 13.5B total / 1.5B active.
- **Prediction target depends on representation**: for RAE(SigLIP 2), **x-pred > v-pred** on generation; for VAE(FLUX.1), **v-pred** is better (x-pred makes text PPL spike as experts shrink — x-pred is unstable for low-dim latents whose rank exceeds the shrinking expert dim).
- **Sparsity**: fix 16 active experts (hidden 512), grow total pool **32 → 1008** (active ratio 50% → 1.6%). Both modalities keep improving at fixed compute; RAE's diffusion loss keeps dropping while VAE's saturates (semantic reps benefit more from sparsity).
- **Shared experts** (Table 2, 256 total / 16 active): **per-modality shared expert** (one always-on text expert + one always-on vision expert, plus top-15 routed) beats both *no-shared* and a *global* shared expert — DCLM PPL **14.785**, Notes **27.161**, Diff. loss **0.483**, GenEval **0.367**.

**Emergent expert specialization (§6.2).** Analyzing a 13.5B MoE (1.5B active, $G{=}16$, 256 experts, SigLIP 2, x-pred, 1T tokens): despite a load-balancing loss pushing uniform use, **far more experts specialize to text than to vision** (consistent with the scaling asymmetry — language is parameter-hungry). Depth pattern: **early layers text-dominated, later layers more vision/multimodal → a "separate-then-integrate" strategy.** Vision experts are **time-invariant** (CV ≈ 0.15 across 10 diffusion-timestep bins — *no* high-noise-vs-fine-detail specialization, unlike Wan). And **visual understanding & generation share the same experts** (routing correlation r ≥ 0.90 across all layers) — a genuinely *unified* visual representation.

### Training procedure

- **From scratch, controlled compute** within each ablation family; evaluated at end of pretraining **without instruction tuning** (except VQA, which gets 1 epoch of Cambrian-7M finetuning).
- Data mixing: each batch blends text-only, video-only, paired image–text, and action-conditioned examples; weighted objective with $\lambda_{\text{flow}}=3\lambda_{LM}$.
- **Classifier-free guidance**: conditioning randomly dropped **10%** of the time during training; inference uses a **fixed CFG = 3.0** (untuned).
- FLOPs accounting for scaling: $6ND \approx C$ for dense; **active** params $N_{\text{active}}$ for MoE (estimated via torchtitan).

### Inference / sampling

- **Text**: standard autoregressive next-token decoding.
- **Vision**: denoise with a **25-step Euler sampler**, then map the final latent to pixels with a **pretrained decoder** (SD-VAE / FLUX.1 off-the-shelf decoders for VAEs; the RAE decoder for SigLIP 2 / DINOv2-L / WebSSL-L).
- **World-model rollout**: feed 4 context frames + a text action, autoregressively predict future frames (per-frame independent noise from Diffusion-Forcing-style training makes this possible).

## Experimental setup

- **Data sources**: (i) web text — **DCLM**; (ii) raw videos — YouTube + curated (Kinetics, SthSth, etc.) at **1 FPS**; (iii) paired image–text — **MetaCLIP** + in-house **Shutterstock (SSTK)**; (iv) action-conditioned navigation trajectories formatted as **I + T → I**, from **NWM** (Bar et al. 2025) + annotated video.
- **Metrics**: text **PPL** (held-out DCLM + OOD in-house "Notes"); **diffusion loss** (held-out CC12M); image generation on **DPGBench** and **GenEval**; **VQA** averaged over the 16 Cambrian benchmarks (after 1-epoch Cambrian-7M finetune); world-modeling **ATE / RPE** (NWM protocol); knowledge-in-generation via **WISE**; **FID** for generation quality.
- **Baselines**: text-only LM, T2I-only generator, dense vs MoT vs MoE, dual-encoder vs single-encoder, and the Transfusion starting point.

## Key results

**Data (§4).**
- **Vision barely hurts text.** Text+Video matches/beats the text-only DCLM PPL; Text+MetaCLIP is the *worst* (caption distribution shift, not vision itself). Adding video+action only slightly degrades text.
- **I/T pairs are essential** for acquiring understanding *and* generation.
- **Caption-distribution gap drives text degradation** (Table 1, cosine distance to DCLM): MetaCLIP **0.196**, MetaCLIP-Recap **0.286**, SSTK **0.215**; larger distance ⇒ worse PPL. Fix: **decouple by objective** — MetaCLIP for image→text, SSTK for text→image — capturing the strengths of both.
- **Synergy**: adding text to a fixed vision budget improves diffusion loss + GenEval; **20B VQA + 80B heterogeneous data beats a 100B VQA-only baseline** (5× less in-domain data), and even unlabeled video helps VQA.

**World modeling (§5).** Actions encoded as **plain text numeric strings** (no special adapters, no architecture change). 50B-NWM + 50B-multimodal beats 50B *and* 100B NWM-only; **pure video gives the biggest gain**. With a fixed 200B budget, performance **saturates at ~1% in-domain NWM data**. Free-form natural-language actions (e.g. *"get out of the shadow!"*) work **zero-shot / OOD**, alongside predefined WASD controls.

**Architecture stacking (§6.3), progressive build from a Transfusion baseline:**
| Step | change | text PPL | DPG |
|---|---|---|---|
| baseline | shared FFN | 15.93 | 0.45 |
| +modality-specific FFN | LMFusion-style split | 15.13 | 0.47 |
| +SigLIP 2 encoder | single semantic enc. (beats dual-encoder) | 15.06 | 0.57 |
| +MoE | data-driven separation (> dense, > MoT) | **12.49** | 0.63 |
| +x-pred | over v-pred | — | **0.65** |

- **WISE** (world-knowledge in generation): semantic encoders beat VAE-based by **3–4×**; MoE highest overall.

**Scaling laws (§7), IsoFLOP, fitting $N_{\text{opt}} \propto C^a$, $D_{\text{opt}} \propto C^b$, $a+b=1$:**
- **Dense — Language** $a{\approx}0.47, b{\approx}0.53$ (Chinchilla-like, balanced). **Vision** $a{\approx}0.37, b{\approx}0.63$ (**data-hungry**). The exponent gap means no single compute-optimal trend governs both.
- Required **vision:language data ratio grows as $O(N^{0.57})$**: from a 1B baseline, **14× more vision data at 100B params, 51× at 1T** — a real dilemma at scale.
- Dense **compute efficiency** $L(C)=A\,C^{-\alpha}+E$: multimodal **matches** text-only (Notes 23.7 vs 23.8; DCLM 13.3 vs 13.6 @$10^{21}$ FLOPs) and **beats** T2I-only (DPG 0.622 vs 0.598; FID 39.3 vs 41.5).
- **MoE (sparsity 16)** — Language $a{\approx}0.41,b{\approx}0.59$; Vision $a{\approx}0.36,b{\approx}0.64$. **MoE halves the parameter-scaling exponent gap (0.10 → 0.05)** by pushing language toward a more data-hungry regime. MoE compute efficiency @$10^{21}$: DCLM PPL 12.3 vs 12.0 (MoE text-only); FID 39.2 vs 39.8 (MoE T2I-only) — a single MoE model matches both unimodal specialists.

## Ablations

The whole paper is essentially one big ablation program: visual encoder family (§3), data-mixture composition + I/T source decoupling (§4.1–4.2), MoE granularity / sparsity / shared-expert / prediction-target sweeps (§6.1), specialization analysis (§6.2), and the progressive stacking table (§6.3). Most load-bearing: **modality-specific FFN → SigLIP-2-RAE → MoE → x-pred** is the path from PPL 15.93/DPG 0.45 to PPL 12.49/DPG 0.65.

## Limitations

(Paper's own + author flags.) Conclusions stem from a **from-scratch** setting — MoT has proven effective specifically when *finetuning* pretrained dense LLMs, so the dense-vs-MoT-vs-MoE verdict may not transfer to the warm-start regime. The scaling asymmetry forces a real compromise (under-train vision or over-train language) that current compute budgets can't fully resolve at 1T scale. Raw-pixel generation still trails semantic encoders. CFG fixed at 3.0 untuned. [analyst] The "world modeling" is evaluated on navigation (NWM) with LPIPS-based planning — narrow relative to the broad "world model" claim; physical-dynamics fidelity beyond navigation is untested here.

## Why it matters [analyst's view]

1. **The scaling asymmetry is the keeper result.** "Vision is more data-hungry than language" with *measured* exponents (dense vision $b\approx0.63$ vs language $b\approx0.53$) means native-multimodal scaling plans must budget data **per modality**, not per parameter — a different mental model from one-Chinchilla-curve-for-all-tokens. Slots directly next to [[papers/yang-2026-replaid-continuous-diffusion]] and the Chinchilla-revision theme in [[topics/scaling-laws]]: headline scaling numbers are *protocol-bound*.
2. **MoE reframed from compute trick to structural prior.** The paper's strongest conceptual move: sparsity is "not efficiency alone — it provides the structural flexibility for modalities with fundamentally different scaling behaviors to coexist." MoE *measurably* drags language's data exponent up (0.53→0.59) toward vision's, shrinking the asymmetry. That's MoE doing *statistical* work, not just FLOP-saving.
3. **Emergent world modeling vs explicit world models.** Actions as text + no architecture change + ~1% in-domain data is a strong "scale dissolves the problem" claim — a direct philosophical contrast with [[papers/maes-2026-leworldmodel]] (explicit JEPA world model, two losses) and the [[topics/world-models]] line generally. Worth weighing against [[papers/joseph-2026-physics-video-world-models]], which finds physics is encoded in *distributed, non-factorised* form — consistent with "emerges from general training," but cautions against over-reading the capability.
4. **RAE collapses the dual-encoder orthodoxy.** A frozen *semantic* encoder + diffusion decoder doing both understanding and generation (with experts shared across the two, r≥0.90) is a tidy unification, and the "separate-then-integrate" depth pattern is a clean interpretability nugget.

The bigger bet (FAIR/Meta + NYU, LeCun on it): **the path to world models is scaling unified multimodal pretraining**, not task-specific objectives — using the "endless" visual signal as the data resource once text runs out.

## Open questions / things to verify

- Is the vision data-hunger **intrinsic** to the modality, or an artifact of current visual-data *quality* (web-scrape vs clean caption) and of using SigLIP-2 latents specifically? Would discrete-token vision change the exponents?
- Does the asymmetry hold under **warm-started** (LLM-init) training, where most production multimodal models actually live?
- Does "world modeling emerges" survive harder dynamics than navigation (contact physics, long-horizon causality)?
- The per-modality shared-expert win is small (PPL 14.802 → 14.785) — is it robust across scale, or noise?
- x-pred-beats-v-pred is representation-dependent; is there a principled rule beyond "high-dim → x-pred"?

## Connections

- Builds on: **Transfusion** (Zhou et al. 2025a); **RAE** (Zheng et al. 2026; Tong et al. 2026); flow matching (Lipman 2023, Liu 2023); Diffusion Forcing (Chen 2024a).
- Topic MOCs: [[topics/world-models]], [[topics/scaling-laws]], [[topics/generative-models]]
- Related papers:
  - [[papers/maes-2026-leworldmodel]] — explicit JEPA world model from pixels; structural alternative to the "emerges from scale" framing here.
  - [[papers/joseph-2026-physics-video-world-models]] — interpretability of physics in video models; tests how much to trust "emergent world modeling."
  - [[papers/yang-2026-replaid-continuous-diffusion]] — continuous vs discrete diffusion LM scaling; same "scaling numbers are protocol-bound" theme.
  - [[papers/ding-2024-diffusion-world-model]] — diffusion as one-shot trajectory predictor; contrast in how diffusion meets world modeling.
- Author indices: [[authors/shengbang-tong]]

## Selected quotes

> "We provide empirical clarity through controlled, from-scratch pretraining experiments, isolating the factors that govern multimodal pretraining without interference from language pretraining." — Abstract

> "Through IsoFLOP analysis, we compute scaling laws for both modalities and uncover a scaling asymmetry: vision is significantly more data-hungry than language." — Abstract

> "This suggests that sparsity plays a deeper role in multimodal models than efficiency alone—it provides the structural flexibility for enabling modalities with fundamentally different scaling behaviors to coexist." — §9 Discussion

> "The experts activated for captioning are the same experts activated for denoising." — §6.2 (paraphrase of the r ≥ 0.90 understanding/generation routing correlation)
