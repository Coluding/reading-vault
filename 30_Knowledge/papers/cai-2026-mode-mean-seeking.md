---
type: paper
title: Mode Seeking meets Mean Seeking for Fast Long Video Generation
authors:
  - Shengqu Cai
  - Weili Nie
  - Chao Liu
  - Julius Berner
  - Lvmin Zhang
  - Nanye Ma
  - Hansheng Chen
  - Maneesh Agrawala
  - Leonidas Guibas
  - Gordon Wetzstein
  - Arash Vahdat
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2602.24289
rw_id: 01ktspg3hj0qw41frj8wwe8jmh
topics:
  - video-generation
  - generative-models
  - distillation
  - flow-matching
priority: high
read_state: read
relevance: high
added: 2026-06-10
last_updated: 2026-06-17
comment: Need to understand why the KL loss is enough to discard the long video generation head and the fact that we actually remov ethe long video generation head completely during inference is astonishing to me
---

# Mode Seeking meets Mean Seeking for Fast Long Video Generation

## TL;DR

The paper attacks the "fidelity–horizon gap" in scaling video generation from seconds to minutes: long, coherent video data is scarce, so models trained on it lose the sharp local realism of short-video generators. The method (project page "mmm", Decoupled Diffusion Transformer / DDT) decouples two objectives onto two lightweight velocity heads sharing one long-context encoder: a **mean-seeking** Flow Matching (FM) head trained via supervised flow matching (SFT) on scarce real long videos to learn minute-scale narrative coherence, and a **mode-seeking** Distribution Matching (DM) head that aligns every sliding window of the student's rollout to a frozen short-video teacher via a reverse-KL DMD/VSD distillation objective — without requiring any extra long-video data. At inference the FM head is discarded and only the few-step DM head generates, unlocking fast minute-scale (30 s) video. On a VBench-Long protocol over 200 long-video prompts with a Wan 1.3B backbone, the method achieves the best overall scores (e.g. VLM Consistency 75.42, Aesthetic Quality 0.5735, Image Quality 0.6982) versus SFT-only and teacher-only (CausVid, Self-Forcing, InfinityRoPE) baselines.

## Context & motivation

Modern video diffusion/transformer models excel at short clips (a few seconds) because seconds-long clips are abundant at web scale (e.g. WebVid-style video–text datasets). Scaling to minutes "breaks this recipe": high-quality long-form videos with sustained events and context are far scarcer, more heterogeneous, and more expensive to curate.

A common practice borrowed from image generation is to train one model on a "soup" of videos of varying lengths, hoping it will interpolate across temporal horizons the way image models interpolate across spatial resolutions. The paper argues (Fig. 1) this assumption is **fundamentally flawed**: a 1024×1024 image is a higher-resolution *interpolation* of a 256×256 image (same underlying patch distribution), but a one-minute video is not an interpolation of a 5-second one — it is a temporal **extrapolation** that adds new events, causal chains and narrative structure, "in some sense similar to panorama generation," requiring much stronger context understanding.

This induces a critical failure mode: even when trained on mixed-duration data, a model that learns to produce longer sequences does so at the cost of losing the sharp local dynamics characteristic of expert short-clip teachers — outputs look softer, less detailed, less "alive." The strongest prior over realistic short-timescale dynamics lives in a heavily-refined short-video generator, while long-video training operates in a data-constrained, expensive regime. The contribution is a training paradigm that **decouples local fidelity from long-term coherence**: long videos supervise global structure; a frozen expert short-video teacher provides a local-realism prior on every sliding window via mode-seeking reverse-KL (implemented as DMD-style distillation). Because the teacher alignment runs through DMD, the resulting DM head doubles as a few-step sampler enabling fast inference.

## Method

### Problem formulation

Work in the latent space of a video VAE with rectified-flow parameterization. Let $x_0 \in \mathbb{R}^{T \times H \times W \times C}$ be a clean video latent (short or long), and $z \sim \pi \triangleq \mathcal{N}(0, I)$ a prior sample. Construct a deterministic noising path

$$x_t = I_t(x_0, z) \triangleq (1-t)\,x_0 + t\,z, \quad t \in [0,1] \quad (1)$$

which induces a generative ODE

$$\frac{dx_t}{dt} = -u(x_t, t), \quad x_1 \sim \pi \quad (2)$$

where $u : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is the marginal velocity field and $d$ the latent dimensionality. Standard flow matching trains $u_\theta$ to approximate $u$ via

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{x_0, z, t}\big[\,\|u_\theta(x_t, t, c) - u(x_t, t \mid x_0, z)\|_2^2\,\big] \quad (3)$$

with conditional target $u(x_t, t \mid x_0, z) = x_0 - z$, and $c$ the conditioning (e.g. text prompt). Sampling solves (2) backward from $t=1$ to $t=0$ using $u_\theta$.

A pre-trained 5-second expert short-video teacher $u_{\text{teacher}}$ (a DiT-style flow model) is assumed available: its velocity field $u_{\text{teacher}}(x_t, t, c)$ can be queried at arbitrary $(x_t, t, c)$, but its original short-clip training data and post-training recipe need not be accessible.

### Sliding-window view of long videos

Given a long-video latent $x_0^{\text{long}}$ of $T_{\text{long}}$ frames and window length $L$ (≈ 5 s of frames), define

$$\text{crop}_k(x_0^{\text{long}}) \in \mathbb{R}^{L \times H \times W \times C} \quad (4)$$

the contiguous $L$-frame segment starting at index $k$, $k \in \{0, \dots, \lceil T_{\text{long}}/L \rceil - 1\}$. Let $q_\Phi^{(k)}$ be the marginal distribution over such windows induced by the long-video student (parameters $\Phi$) and $p_{\text{teacher}}$ the short-video teacher distribution. The goal is to regularize

$$\mathcal{L}_{\text{seg}}(\Phi) = \mathbb{E}_k\big[\,D_{\text{KL}}(q_\Phi^{(k)} \,\|\, p_{\text{teacher}})\,\big] \quad (5)$$

i.e. every sliding window should match the short-video teacher distribution. Because it uses the **reverse** KL $D_{\text{KL}}(q\|p)$, this objective is **mode-seeking**: it pushes the student to concentrate mass on the teacher's high-fidelity modes rather than average over them. The short teacher acts as a local realism critic on windows sampled on-policy from long rollouts, beyond its native horizon.

### Core idea

The reverse-KL teacher term (5) is mode-seeking, while the flow-matching SFT objective (3) is a conditional-mean predictor and thus **mean-seeking**. Forcing both signals through a single velocity predictor yields gradient interference (averaging-under-ambiguity vs. committing-to-modes). The fix: instantiate the long-video student as a **decoupled** architecture with a shared feature encoder feeding two lightweight velocity heads (one per objective), inspired by DDT (Wang et al., 2025c).

### Architecture / algorithm

A shared long-context **condition encoder** $E_\phi$ maps a noisy long-video latent, timestep and conditioning to a unified spatiotemporal representation:

$$h_t = E_\phi(x_t^{\text{long}}, t, c) \quad (6)$$

$E_\phi$ is a video diffusion transformer with full-range temporal dependencies (full attention) — the shared backbone. Two lightweight transformer **decoder heads** read velocities from $h_t$:

$$u_\theta(x_t^{\text{long}}, t, c) = D_\theta^{\text{FM}}(h_t, t, c) \quad (7)$$
$$v_\psi(x_t^{\text{long}}, t, c) = D_\psi^{\text{DM}}(h_t, t, c) \quad (8)$$

- The **Flow Matching head** $D_\theta^{\text{FM}}$ parameterizes the student's global velocity $u_\theta$, trained and sampled on ground-truth long videos via (3).
- The **Distribution Matching head** $D_\psi^{\text{DM}}$ is a **few-step generator** distilled to align local windows to the teacher via DMD/VSD.

Sharing $E_\phi$ but decoupling heads gives: (i) the long-context representation $h_t$ is learned and reused across both objectives, and (ii) short-video generation capability is distilled from the teacher without forgetting it on scarce long data.

### Derivations / why it works

**Reverse-KL gradient on noised windows (DMD/VSD surrogate).** Directly evaluating (5) is intractable; the paper follows the DMD/VSD result that for diffusion/flow models, the gradient of a reverse-KL between student and teacher can be written as an expectation over the student distribution involving the difference between the teacher's score/velocity and the student's own "fake" score/velocity on the same noisy state. Let $\hat{q}_{\Phi,t}^{(k)}$ be the distribution of the noised window $\hat{x}_t^{(k)} = (1-t)\hat{x}_0^{(k)} + t\,\epsilon$ where $\hat{x}_0^{(k)}$ is a student sample and $\epsilon \sim \mathcal{N}(0, I)$ is fresh noise. Using the linear interpolation in (1) (with $\alpha_t = 1-t$), the window-level gradient surrogate is

$$\widehat{\nabla}\mathcal{L}_{\text{seg}} = \mathbb{E}_{t,k}\Big[\lambda(t)\big(v_{\text{fake}}(\hat{x}_t^{\text{long}}, t, c) - u_{\text{teacher}}(\hat{x}_t^{(k)}, t, c)\big)^\top \nabla \hat{x}_0^{\text{long}}\Big] \quad (9)$$

where $\hat{x}_0^{(k)}$ is the generated student window, $v_{\text{fake}}$ is a fake-score estimator trained on the student's own window predictions $\hat{x}_0^{(k)}$ via score-matching between student updates, $\lambda(t)$ absorbs the standard DMD/VSD weighting (incl. score-to-velocity scaling), $u_{\text{teacher}}$ is the short-video teacher velocity query, and $v_\psi^{(k)}$ is the student's mode-seeking head velocity. The term $(v_{\text{fake}} - u_{\text{teacher}})$ is treated as **stop-gradient**, backpropagating only through $\hat{x}_0^{\text{long}}$.

**Cropping the mode-seeking head.** The generated window is obtained by cropping the DM head's output on the noised long latent:

$$x_t^{\text{long}} = (1-t)\,x_0^{\text{long}} + t\,\epsilon \quad (10)$$
$$h_t = E_\phi(x_t^{\text{long}}, t, c) \quad (11)$$
$$v_\psi^{\text{long}}(x_t^{\text{long}}, t, c) = D_\psi^{\text{DM}}(h_t, t, c) \quad (12)$$
$$v_\psi^{(k)}(x_t^{\text{long}}, t, c) = \text{crop}_k\big(v_\psi^{\text{long}}(x_t^{\text{long}}, t, c)\big) \quad (13)$$

with $\hat{x}^{(k)} = \text{crop}_k(\hat{x}_t^{\text{long}})$, $\epsilon \sim \mathcal{N}(0,I)$. The teacher term $u_{\text{teacher}}(\hat{x}_t^{(k)}, t, c)$ is evaluated on the cropped $K$-frame window using the short-video teacher. (Sliding-Window DMD is non-trivial to implement on modern video generators; details deferred to supplementary §C.)

**SFT flow-matching anchor on long videos.** Distribution matching alone cannot teach global coherence (the short teacher does not model long-range structure by design). The FM head $u_\theta$ is trained with a supervised flow-matching objective on full-length real long videos. With $x_0^{\text{long}} \sim p_{\text{long}}$, global prior $z^{\text{long}} \sim \pi$, and $x_t^{\text{long}}$ from (1):

$$\mathcal{L}_{\text{SFT}}(\phi, \theta) = \mathbb{E}_{x_0^{\text{long}}, z^{\text{long}}, t}\big[\,\|u_\theta(x_t^{\text{long}}, t, c) - (x_0^{\text{long}} - z^{\text{long}})\|_2^2\,\big] \quad (14)$$

Gradients flow through both $E_\phi$ and the FM head $D_\theta^{\text{FM}}$, anchoring the student's global velocity to real long-video trajectories (correct long-horizon temporal dependencies and narrative structure).

### Training procedure

The joint objective combines supervised long-video flow matching on the global FM head with local reverse-KL alignment on the DM head:

$$\mathcal{L}_{\text{total}}(\phi, \theta, \psi) = \mathcal{L}_{\text{SFT}}(\phi, \theta) + \lambda_{\text{seg}}\,\mathcal{L}_{\text{seg}}(\phi, \psi) \quad (15)$$

with scalar weight $\lambda_{\text{seg}}$, and $\mathcal{L}_{\text{seg}}$'s effect injected via the surrogate (9). The shared encoder receives gradients from both terms; the heads only from their own signal:

$$\nabla_\phi \mathcal{L}_{\text{total}} = \nabla_\phi \mathcal{L}_{\text{SFT}} + \lambda_{\text{seg}}\,\widehat{\nabla}_\phi \mathcal{L}_{\text{seg}} \quad (16)$$
$$\nabla_\theta \mathcal{L}_{\text{total}} = \nabla_\theta \mathcal{L}_{\text{SFT}} \quad (17)$$
$$\nabla_\psi \mathcal{L}_{\text{total}} = \lambda_{\text{seg}}\,\widehat{\nabla}_\psi \mathcal{L}_{\text{seg}} \quad (18)$$

This realizes the decoupling: long-video supervision updates the global FM head, local short-video teacher distribution matching is handled by the DM head, and both update the shared representation. Each step uses two minibatches: (1) a batch of real long videos to compute $\mathcal{L}_{\text{SFT}}$ and update $(\phi, \theta)$; (2) a batch of on-policy student rollouts to compute and apply the DMD/VSD surrogate (9) to $(\phi, \psi)$. The teacher's reverse-KL alignment runs DMD over 5 steps on the same noisy state. Base models are Wan 1.3B (used as both student and teacher for fair comparison) and Wan 14B (qualitative results). The Sekai dataset is acknowledged (Li et al., 2025c).

### Inference / sampling

At inference the FM head is **discarded**; long videos are generated end-to-end using only the DM head $v_\psi$, supervised solely on sliding windows (similar to APT2 / Lin et al. 2025) but able to generate long videos bidirectionally. The shared encoder guarantees any sliding window of the generated long video resides in the same local distributional modes as the short-video teacher, while long-range coherence comes from $\mathcal{L}_{\text{SFT}}$-shaped representation. The DM head is a **few-step** generator, so inference is fast minute-scale generation. The method is orthogonal to causal AR models and can be combined with them (noted as future work).

## Experimental setup

- **Base models:** Wan 1.3B (all quantitative + qualitative comparisons, used as both student and teacher) and Wan 14B (qualitative only).
- **Test set / metric:** 200 prompts describing long videos/events; generate 30-second videos. VBench-Long protocol — Subject Consistency, Background Consistency, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality. Plus a "VLM Consistency" score from Gemini-3-Pro (advanced MLLM taking video inputs).
- **Baselines:** (1) **Long-context SFT** (LongSFT) — basic long-tuning on collected long clips, 50 NFE; (2) **Mixed-length SFT** (MixSFT) — industrial-style joint training over varying lengths, 50 NFE; (3) autoregressive teacher-distillation methods **CausVid** (Yin et al. 2025) and **Self-Forcing** (Huang et al. 2025b) using CausVid's extrapolation, 4 NFE; (4) **InfinityRoPE** (Yesiltepe et al. 2026), a Self-Forcing extrapolation method, 4 NFE. The method runs at 4 NFE.

## Key results

Table 1 (↑ = higher better; *summarized headline numbers*):

| Method | NFE | Subject Cons. | Bg Cons. | Motion Smooth. | Dynamic Deg. | Aesthetic | Image Qual. | VLM Cons. |
|---|---|---|---|---|---|---|---|---|
| Long-context SFT | 50 | 0.9685 | 0.9533 | 0.9866 | 0.9375 | 0.4973 | 0.6303 | 77.28 |
| Mixed-length SFT | 50 | 0.9667 | 0.9541 | 0.9874 | 0.8906 | 0.5467 | 0.6683 | 74.63 |
| CausVid | 4 | 0.9736 | 0.9614 | 0.9789 | 0.8594 | 0.6044 | 0.6305 | 39.30 |
| Self-Forcing | 4 | 0.9489 | 0.9451 | 0.9805 | 0.9063 | 0.5556 | 0.6278 | 37.60 |
| InfinityRoPE | 4 | 0.9689 | 0.9573 | 0.9812 | 0.7188 | 0.5342 | 0.6871 | 68.61 |
| **Ours** | 4 | 0.9682 | 0.9548 | 0.9863 | 0.9453 | **0.5735** | **0.6982** | **75.42** |

The authors flag that AR-based methods get over-saturated (CausVid) and static (InfinityRoPE), which artificially boosts some consistency metrics — yet AR methods collapse on VLM Consistency (CausVid 39.30, Self-Forcing 37.60), exposing poor true long-range coherence. Ours is the best overall: top Aesthetic and Image Quality, near-top Dynamic Degree, and competitive VLM Consistency at only 4 NFE (vs. 50 NFE for the SFT baselines). Qualitatively (Figs. 3–4) SFT-only methods preserve layout but wash out fine texture / soften edges / collapse subjects; teacher-only AR methods degrade over long range (CausVid/Self-Forcing) or generate near-static "sink"-anchored content (InfinityRoPE).

## Ablations

Table 2 removes three ingredients (↑):

| Variant | Consistency | Motion | Quality |
|---|---|---|---|
| No DDT dual heads | 0.9427 | 0.9449 | 0.5298 |
| No Sliding-window DMD | 0.9604 | 0.9621 | 0.6075 |
| No SFT | 0.9579 | 0.9690 | 0.5862 |
| **Full Model** | **0.9615** | **0.9685** | **0.6359** |

- **No DDT dual heads** (single velocity predictor with both signals) gives the **largest drop across all metrics**, confirming the gradient-interference motivation — the two objectives must not share one head.
- **No Sliding-window DMD** degrades into SFT-only: lower quality (washed local realism).
- **No SFT** (teacher alignment only) keeps competitive quality but **worse global consistency** — the short teacher is "blind" to minute-scale narrative structure and cannot substitute for long-video supervision.

Conclusion: SFT supplies the long-horizon anchor, teacher DM restores short-horizon fidelity, and the decoupled dual-head architecture is essential to combine them.

## Limitations

- The dual-head decoupling adds architectural complexity and requires balancing $\lambda_{\text{seg}}$ between competing objectives.
- Quantitative evidence is on a Wan **1.3B** backbone; 14B is only qualitative. [analyst's view] No reported per-metric numbers at 14B.
- The teacher is fixed at ≈5 s; quality of the local realism prior is upper-bounded by that teacher. [analyst's view]
- Sliding-Window DMD on modern video models is "non-trivial to implement" — details are deferred to supplementary §C, so the note can't fully reconstruct the windowing/score-estimator mechanics. _Partially not addressed by the main text._
- Long here means 30 s in the quantitative eval; "minute-scale" claims rest largely on qualitative figures. [analyst's view]

## Why it matters [analyst's view]

This is a clean instance of the broader distillation pattern where reverse-KL (mode-seeking, DMD/VSD) and forward-objective (mean-seeking, flow/score matching) are deliberately *separated by architecture* rather than blended in one loss — the gradient-interference framing is the conceptual core and generalizes beyond video. The data argument (video length is extrapolation, not interpolation, unlike image resolution) is a sharp and quotable reframe of why "train on a soup of lengths" underdelivers. The fact that the inference path is the *few-step DM head only* means the coherence supervision is amortized entirely into the shared encoder — you get long-horizon structure for free at sampling time without paying the FM head's cost. This bridges the "fast distilled sampler" line (DMD/Self-Forcing/CausVid) with the "long-horizon world-model video" line, and the orthogonality-to-AR claim suggests it could be stacked on causal rollout methods.

## Open questions / things to verify

- Exact $\lambda_{\text{seg}}$, optimizer, learning rate, schedule, step counts — not in the main text; check supplementary.
- How is $v_{\text{fake}}$ (fake-score estimator) trained and updated in lockstep — frequency, separate optimizer? (DMD2-style two-time-scale?)
- Does using the *same* Wan 1.3B as both teacher and student bias the comparison favorably vs. baselines? How does a stronger/different teacher change results?
- 14B quantitative numbers and behavior beyond 30 s toward true minute-scale.
- Robustness of sliding-window cropping at window boundaries — seam artifacts.

## Connections

- Builds on: DMD / VSD distillation (Yin et al. 2024a/b; Wang et al. 2023b), rectified flow / flow matching (Lipman et al. 2023; Liu et al. 2023), Decoupled Diffusion Transformer / DDT (Wang et al. 2025c), Wan video backbone (Wang et al. 2025a).
- Contrasts with: CausVid (Yin et al. 2025), Self-Forcing (Huang et al. 2025b), InfinityRoPE (Yesiltepe et al. 2026) — teacher-only AR distillation baselines it outperforms on long-horizon coherence.
- Related vault notes (mode/mean-seeking & flow distillation territory): [[papers/pao-huang-2026-flux-matching]], [[papers/bartosh-2026-dual-rate-diffusion]], [[papers/yang-2026-replaid-continuous-diffusion]]
- Topic MOCs: [[topics/video-generation]], [[topics/generative-models]], [[topics/distillation]], [[topics/flow-matching]]
- Author indices: [[authors/shengqu-cai]], [[authors/weili-nie]], [[authors/chao-liu]], [[authors/julius-berner]], [[authors/lvmin-zhang]], [[authors/nanye-ma]], [[authors/hansheng-chen]], [[authors/maneesh-agrawala]], [[authors/leonidas-guibas]], [[authors/gordon-wetzstein]], [[authors/arash-vahdat]]

## Selected quotes

> "We argue that this assumption is fundamentally flawed... a one-minute video, however, is not an 'interpolation' of a 5-second video: it is an 'extrapolation' that adds new events, causal chains, and narrative structure." — §1

> "Because we use the reverse KL, $D_{\text{KL}}(q \| p)$, this objective is mode-seeking: it encourages the student to concentrate its mass on the teacher's high-fidelity modes rather than averaging over them." — §3.2

> "By using only the DM head for inference, we unlock fast, few-step long-video inference." — §1 (contributions)
