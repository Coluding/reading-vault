---
type: paper
title: "Inference-time Physics Alignment of Video Generative Models with Latent World Models"
authors: ["Jianhao Yuan", "Xiaofeng Zhang", "Felix Friedrich", "Nicolas Beltran-Velez", "Melissa Hall", "Reyhane Askari-Hemmat", "Xiaochuang Han", "Nicolas Ballas", "Michal Drozdzal", "Adriana Romero-Soriano"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2601.10553
rw_id: 01kvq3ts82zkkyxj0nz60zp35k
topics: [video-generation, world-models, diffusion-models, inference-time-scaling]
priority: high
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

The paper (from FAIR / Meta Superintelligence Labs, dated March 2, 2026) shows that a large share of the physics-implausibility in state-of-the-art video generators is not a pre-training deficit but a *sampling* problem, and reframes fixing it as an **inference-time alignment** task. Their method, **WMReward**, re-purposes the "surprise score" of a latent world model (VJEPA-2) — the cosine mismatch between the world model's predicted future latents and the generated video's actual latents — into a differentiable reward, then searches/steers denoising trajectories toward a reward-tilted distribution. It works across image-to-video (I2V), video-to-video (V2V), and text-to-video (T2V) with three different generators (MAGI-1 24B, a 5B video LDM, Sora2). With the combined guidance+selection sampler ($\nabla$+BoN) it reaches a **62.0% final PhysicsIQ score on V2V, +6.78% over the previous SOTA** (MAGI-1), and won first place in the ICCV 2025 Perception Test PhysicsIQ Challenge with an official **62.64% (+7.42%)**. Crucially, latent-world-model surprise beats VLM judges (Qwen3-VL near chance) and pixel-space MAE (VideoMAE) as a reward, and the gains require **no retraining** of the generator.

## Context & motivation

Video generative models produce visually pleasing clips but routinely violate basic physics (gravity, mass conservation, rigid-body dynamics, fluids), which limits their use as world models and for robotics / autonomous driving. Prior work mostly blamed the pre-training objective (pixel/feature reconstruction) and tried to inject physics during pre-/post-training (e.g. VideoJAM, NewtonGen, distillation from physics foundation models). A second, underexplored line keeps the generator fixed and improves physics *at inference*, but to date was limited to VLM-based prompt rewriting (PhyT2V) or motion planning (VLIPP) on top of motion-controllable generators. Meanwhile, in *image* generation, reward-model search over the generation manifold (Best-of-N over seeds, gradient guidance) is well established. The paper's insight: latent world models such as VJEPA-2 have been shown (Garrido et al. 2025) to develop strong intuitive-physics understanding by predicting in a compressed latent space, so they are ideal *off-the-shelf reward models* for physics plausibility — a use nobody had exploited for video generation.

## Method

### Problem formulation
Given a pre-trained video model with distribution $p(x)$ over videos $x$, sample from a **reward-tilted distribution**
$$p^*(x) \propto w(x)\,p(x),$$
where $w(x)>0$ is a weighting built from a physics-plausibility reward $r(x)$. The two design questions are (1) how to build $r(x)$ from a latent world model, and (2) how to sample from $p^*$ given only the pre-trained $p(x)$ and $r(x)$.

Diffusion / flow-matching preliminaries: these learn the score $\nabla_{x_t}\log p_t(x_t)$ of noised variables $x_t = \alpha_t x_0 + \sigma_t \epsilon$, with $x_0\sim p_{\text{data}}$, $\epsilon\sim\mathcal{N}(0,I)$, $\alpha_0=1,\sigma_0=0$ (so $p_0=p_{\text{data}}$) and $p_T$ approximately Gaussian. Sampling solves the reverse SDE/ODE from $t=T$ to $t=0$.

### Core idea
A world model should predict the future well *only for physically plausible videos*. So the divergence between VJEPA-2's predicted future latents and the generated video's actual future latents ("surprise") is a proxy for implausibility — and being fully differentiable, it can drive both gradient-free selection and gradient guidance.

### Architecture / algorithm
**VJEPA-2 recap.** A context encoder $E_\theta$ embeds frames; a predictor $P_\phi$ reconstructs target representations from masked ones. Trained self-supervised with
$$\mathcal{L} = \lVert P_\phi(\Delta_m, E_\theta(x_{\text{masked}})) - \text{sg}(\bar{E}_\theta(x)) \rVert_1,$$
where $\Delta_m$ are learnable mask tokens marking masked spatiotemporal positions, $\bar{E}_\theta$ is an EMA (target) encoder, and $\text{sg}(\cdot)$ is stop-gradient. Predicting in feature space (not pixels) pushes the model toward dynamics/interaction rather than appearance.

**WMReward (the reward).** Slide a window of length $C+M$ over the generated video ($C$ = context frames, $M$ = prediction horizon). At window position $k$:
- predict future latents from context only: $\hat{z}_k = P_\phi(\Delta_m, E_\theta(x^{k-C+1:k}))$ (Eq. 4), with the $M$ future positions masked;
- encode the *full* window to get the observed latents: $z_k = E_\theta(x^{k-C+1:k+M})$ (Eq. 5).

Extract only the future portions $\hat{z}_k^{\text{fut}}, z_k^{\text{fut}}$ (positions $C+1..C+M$) and average the cosine mismatch:
$$r(x) = \frac{1}{|\mathcal{K}|}\sum_{k\in\mathcal{K}}\bigl(1 - \cos(\hat{z}_k^{\text{fut}}, z_k^{\text{fut}})\bigr) \quad (6)$$
over all valid window positions $\mathcal{K}$. (Note the sign/orientation: the text says videos matching VJEPA's predictions get *higher* reward; Eq. 6 is used as the plausibility signal that guidance ascends.)

### Derivations / why it works
Three sampling schemes instantiate $w(x)$:

**(i) Guidance ($\nabla$).** Use $w(x)=\exp(\lambda r(x))$, $\lambda>0$ a temperature. Under the same noising process, the tilted marginals have score
$$\nabla_{x_t}\log p^*_t(x_t) = \nabla_{x_t}\log p_t(x_t) + \nabla_{x_t}\log \mathbb{E}\!\left[e^{\lambda r(x_0)}\mid x_t\right]. \quad (7)$$
Approximating $\mathbb{E}[e^{\lambda r(x_0)}\mid x_t]\approx e^{\lambda r(\mathbb{E}[x_0\mid x_t])}$ and using **Tweedie's formula**
$$x_{0|t} := \mathbb{E}[x_0\mid x_t] = \tfrac{1}{\alpha_t}\bigl(x_t + \sigma_t \nabla_{x_t}\log p_t(x_t)\bigr), \quad (8)$$
gives the tractable steered score
$$\nabla_{x_t}\log p^*_t(x_t) \approx \nabla_{x_t}\log p_t(x_t) + \lambda\,\nabla_{x_t} r(x_{0|t}(x_t)). \quad (9)$$
So one just adds $\lambda$ times the reward gradient (evaluated at the denoised estimate) to the model's score inside the sampler. The reward gradient requires backprop through VJEPA-2, which is where the cost comes from.

**(ii) Best-of-N (BoN).** Draw $N$ i.i.d. particles from $p(x)$, keep the max-reward one: $x^*=\arg\max r(x^{(i)})$ (Eq. 10). This effectively samples $p^*(x)\propto p(x)[F(r(x))]^{N-1}$ (Eq. 11), $F$ = CDF of $r(x)$ under $p$; i.e. $w(x)=[F(r(x))]^{N-1}$.

**(iii) $\nabla$+BoN.** Generate the $N$ particles *with guidance*, then BoN-select. Weight $w(x)=\exp(\lambda r(x))\cdot[F_\lambda(r(x))]^{N-1}$. This achieves strong tilting without needing large $\lambda$ (where the Tweedie/score approximation degrades), and BoN filters out cases where the guidance approximation was poor. This is the recommended scheme.

**Adaptation to generation paradigms (Appendix A.2).** For *holistic* generators (all frames same noise level, e.g. vLDM, Wan) the WMReward guidance is combined with CFG:
$$\nabla_{x_t}\log p_\lambda(x_t|\text{txt}) = (1-\omega_{\text{txt}})\nabla\log p(x_t) + \omega_{\text{txt}}\nabla\log p(x_t|\text{txt}) - \omega_s\nabla_{x_t} r(x_t|\text{txt}). \quad (12)$$
For *autoregressive* chunk-by-chunk generators (MAGI-1), Eq. 13 adds a term for the previously denoised chunks $x^{<k}_t$ used as context guidance, and uses those chunks as VJEPA context to predict/score the next chunk.

### Training procedure
_No training of the generator or reward model — this is purely inference-time._ VJEPA-2 is used frozen (ViT-giant, input frames $256\times256$, window/context/stride = 16/8/8 as default). Generation hyperparameters (Appendix Table 8): MAGI-1 24B distilled checkpoint on 8×H200, 32 steps, CFG 7.5, VJEPA guidance scale 0.005; vLDM 5B (DiT + spatiotemporal VAE) on 1×H200, 50 DDIM steps, CFG 6.0, VJEPA guidance scale 0.001–0.003. Guidance is applied every few timesteps ("guidance frequency" 1–5), not every step, to save compute.

### Inference / sampling
Sliding-window surprise is computed on the (partly denoised) video; for guidance it is evaluated at the Tweedie estimate $x_{0|t}$, and the reward gradient is added to the score at the chosen timesteps; BoN parallelizes $N$ trajectories and selects at the end.

## Experimental setup

- **Benchmarks / datasets:** PhysicsIQ (I2V and V2V; final score aggregates spatial IoU, spatio-temporal IoU over motion masks, weighted spatial IoU, pixel MSE) — 16 particles; VideoPhy (T2V; VLM-scored Physics Consistency and Semantic Adherence on 344 prompts) — 8 particles.
- **Generators:** MAGI-1 24B (autoregressive), vLDM 5B (holistic DiT), Sora2 (API — guidance unavailable, BoN only).
- **Reward baselines:** VideoMAE (pixel-reconstruction surprise), Qwen2.5-VL-7B and Qwen3-VL-8B (binary physics-plausibility question, positive-token logit as reward).
- **Sampling baselines:** vanilla sampling, plus SMC and SVDD as alternative reward-guided samplers.
- **Extra evals:** VBench visual-quality metrics; a 5-annotator human preference study (198 annotations each for vLDM-I2V and MAGI-1-V2V).

## Key results

- **PhysicsIQ V2V:** MAGI-1 55.22 → **62.0 with $\nabla$+BoN (+6.78)**; official challenge platform reports **62.64 (+7.42)**, first place.
- **PhysicsIQ I2V:** vLDM 27.76 → 33.44 (+5.68); MAGI-1 29.77 → ~36.3 (+6.5); Wan2.2 38.26 → 44.39 (+6.13, BoN); Sora2 42.30 → **46.43 (+4.13, BoN only)** — beating the previous Sora2 SOTA.
- **VideoPhy T2V:** Physics Consistency +8.1% (MAGI-1) and +6.9% (vLDM); but Semantic Adherence *drops* because VJEPA surprise is text-agnostic.
- **Reward comparison:** WMReward >> VideoMAE (small gains) and Qwen VLs (often *negative*, near chance) — supporting that latent-space predictive surprise is the better physics proxy.
- **Human study:** physics-plausibility win rate rises by ~11.4%; e.g. MAGI-1 physics-plausibility accuracy 41.3 → 58.7 on PhysicsIQ, 30.4 → 69.6 on VideoPhy.

## Ablations

- **Scaling particles:** PhysicsIQ score climbs steadily with $N$, biggest jump at $N\le4$, variance shrinks as $N$ grows; guidance sharpens the upper tail of the score KDE (shown $N=1$ vs $N=16$).
- **Sampler comparison (vLDM I2V, Table 6):** BoN and $\nabla$+BoN beat SMC and SVDD at all $N\in\{2,4,8,16\}$ (e.g. at $N=16$: BoN 32.90, $\nabla$+BoN 33.44 vs SMC 29.24, SVDD 28.87). Reason: differentiable VJEPA gradients give more accurate step-wise guidance than SVDD's multi-rollout value estimates; SMC's early noisy rewards cause resampling collapse.
- **VJEPA size / hyperparameters (Table 7):** robust to context $C$, horizon $M$, stride $s$; gains scale with backbone size (ViT-huge → ViT-giant), so stronger world models give better rewards with no generator finetuning. Best config ViT-giant window 32/context 16/stride 16 → 60.78.
- **Perceptual quality (VBench, Table 5):** small *improvements* in imaging/aesthetic quality, motion smoothness, temporal flickering — physics gains don't hurt perceptual quality.
- **Compute cost (Table 4):** BoN is $\times N$ time, $\times 1$ memory per particle (parallelizable); guidance adds $\approx\times 5$ time and $\times 2$–$4$ memory from backprop through VJEPA.

## Limitations

Paper's own: (1) generator quality caps the reachable solution set; (2) VJEPA-2's training data is limited so it misses some phenomena — the authors note weak understanding of material properties like weight/friction; (3) early-diffusion $x_{0|t}$ estimates are blurry, making early reward signals unreliable. Failure modes (Appendix D): abrupt events (fluid overflow, a match igniting a balloon), mirror reflections, siphon effects. Honest-reader flags: the reward is text-agnostic so T2V semantic adherence drops; guidance's $\times5$ compute overhead is large; results depend on the specific VJEPA-2 checkpoint.

## Why it matters [analyst's view]

This is a clean demonstration that **world models and generators are complementary at inference time**: a frozen predictive latent model (VJEPA-2) can score and steer a frozen generator without any joint training, turning "physics understanding" into a plug-in reward. It's a strong data point for the LeCun-style thesis that JEPA-style predictive latents capture dynamics better than pixel reconstruction or VLM judgment — the reward-comparison table is almost an indictment of using VLMs as physics judges. Practically it slots directly into the inference-time-scaling paradigm (BoN/guidance as compute-for-quality trade), and the $\nabla$+BoN framing generalizes to any differentiable off-the-shelf reward. The obvious next step the paper opens is *text-conditioned or compositional* world-model rewards to recover the semantic-adherence loss, and better/faster search to handle blurry early-step estimates. Connects in the vault to world-models and inference-time-scaling.

## Open questions / things to verify

- Sign convention of Eq. 6 vs. the stated "higher reward when matching" — worth checking against the code (facebookresearch/WMReward) since guidance ascends $r$.
- How much of the gain is VJEPA-specific vs. generic to any strong latent world model? (They claim generality but only test VJEPA-2.)
- The $\times5$ guidance overhead vs. simply scaling BoN — is guidance worth it outside the tail?
- Whether the abstract's 62.64% (challenge) vs. 62.0% (paper body) discrepancy is just the official-platform vs. local-eval gap (paper says yes).

## Connections

- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/inference-time-scaling]], [[topics/diffusion-models]]
- Author indices: [[authors/jianhao-yuan]], [[authors/adriana-romero-soriano]], [[authors/nicolas-ballas]]
- Related: builds on VJEPA-2 (Assran et al. 2025) and Garrido et al. 2025 (intuitive physics from SSL); contrasts with pre-/post-training physics injection (VideoJAM, NewtonGen) and with VLM-based inference methods (PhyT2V, VLIPP).

## Selected quotes

> "the shortfall in physics plausibility also stems from suboptimal inference strategies." — Abstract

> "the more its predictions diverge from the generations, the more likely it is that the videos are less physically plausible." — §2.3

> "stronger VJEPA backbones yield better performance without any fine-tuning of the underlying video generator." — §3.4
