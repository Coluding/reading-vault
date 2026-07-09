---
type: paper
title: "The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show"
authors: ["Parsa Esmati", "Somjit Nath", "Katja Hofmann", "Derek Nowrouzezahrai", "Samira Ebrahimi Kahou", "Majid Mirmehdi"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2606.05328
rw_id: 01kvq48pmj1hfp5rfta757dw0b
topics: [video-generation, diffusion-models, world-models]
priority: high
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

This paper asks whether video diffusion models internally *encode* physical structure even when their generated outputs violate physics. The obstacle is access: diffusion models never expose the latent trajectory associated with a *real* video. The authors solve this by approximately inverting the deterministic (flow-matching) sampler — integrating the learned velocity field **backward** from a clean video latent to noise with a single-evaluation explicit Euler/Heun step — recovering the full trajectory of intermediate transformer states. Training linear probes on these states, they find physical plausibility is linearly decodable from diffusion-transformer (DiT) blocks at up to ~81.27% per-video accuracy (WAN-1.3B), **outperforming dedicated representation encoders** V-JEPA 2 (71.36%) and VideoMAE. Crucially the signal is *absent* from the VAE input latents (48–53%, chance) and emerges inside the denoising transformer itself — despite no predictive/physics objective. The signal lives at intermediate depth, persists across the whole reverse trajectory, encodes quantitative scene parameters (initial position/velocity with $R^2 \geq 0.88$), and depends on trajectory *fidelity* rather than endpoint reconstruction.

## Context & motivation

Video diffusion models (Sora, Veo, Cosmos) are increasingly positioned as general-purpose world simulators. But visual realism doesn't require internalizing physics — it only requires producing trajectories statistically similar to training data. Recent evaluations (Kang et al. 2025; Huberman et al. 2026) show these models fail to extrapolate basic mechanics out of distribution, instead mimicking the nearest in-distribution example. Compounding this, they operate in the latent space of a VAE trained purely for *reconstruction*, which is not optimized to capture semantic or physical structure. So the model has neither an explicit objective nor an obvious substrate that would push it to recover physical laws.

The central question: **do modern video generation models encode physical knowledge internally, even when their output fails to capture it?** Prior work approached physical understanding through representation learning — V-JEPA (Bardes et al. 2024) learns predictive latents that distinguish plausible from implausible video (Garrido et al. 2025), and Joseph et al. 2026 localized these signals within model depth. But those are *predictive* encoders; whether comparable structure exists in *generative* diffusion models was open, blocked by the lack of access to latent trajectories for real videos.

## Method

### Problem formulation
Given a real video with known physical plausibility, obtain the diffusion model's internal activations *along the latent trajectory it would associate with that video*, then decode physical plausibility (and quantitative physical variables) from those activations via linear probes.

### Core idea
Recover the latent trajectory by running the model's own sampler **in reverse**: encode the video with the model's VAE to a clean latent, then integrate the forward generative ODE backward to noise, recording every intermediate state and attention map. Probe those states.

### Architecture / algorithm
**Flow-matching preliminaries.** A flow model learns a time-dependent velocity field $v_\theta$ transporting a prior $Z_0 \sim \mathcal{N}(0,I)$ to data $Z_1 \approx$ data, via the forward ODE
$$\frac{d\mathbf{Z}_t}{dt} = v_\theta(\mathbf{Z}_t, t), \quad t \in [0,1], \tag{1}$$
discretized with a one-step Euler integrator,
$$\mathbf{Z}_{t_{k+1}} = \mathbf{Z}_{t_k} + \Delta t_k \cdot v_\theta(\mathbf{Z}_{t_k}, t_k), \quad \Delta t_k = t_{k+1}-t_k. \tag{2}$$
Each velocity evaluation is a full forward pass through the transformer backbone — the computation whose internal states they examine.

**Reverse sampling.** Encode the video with the VAE to $Z_1 = \mathcal{E}(V)$ and integrate Eq. 1 from $t=1$ back to $t=0$. The *exact* reverse of Euler is the implicit step
$$\mathbf{Z}_{t_k} = \mathbf{Z}_{t_{k+1}} - \Delta t_k \cdot v_\theta(\mathbf{Z}_{t_k}, t_k), \tag{3}$$
where the unknown appears on both sides inside the nonlinear $v_\theta$, requiring an iterative solver at every step (≈10× cost). Instead they use an **explicit** approximation, evaluating the velocity at the known endpoint:
$$\mathbf{Z}_{t_k} = \mathbf{Z}_{t_{k+1}} - \Delta t_k \cdot v_\theta(\mathbf{Z}_{t_{k+1}}, t_{k+1}), \tag{4}$$
one network evaluation per step, matching forward-sampling cost. Resampling the recovered $Z_0$ forward reconstructs the original video up to minor artifacts, confirming the recovered internal states traverse video-preserving trajectories (error bounded in Appendix A). Linear probes (Alain & Bengio) are trained on each transformer block's outputs to predict physical plausibility.

**Block intervention (causal).** To ask *which* blocks causally carry the signal, they corrupt each block's output during generation. For block output $\mathbf{h}$:
$$\tilde{\mathbf{h}} = \mathbf{h} + \alpha \cdot \sigma(\mathbf{h}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I), \tag{5}$$
where $\sigma(\mathbf{h})$ is the per-token feature std and $\alpha$ the intervention strength (scaling noise to local activation magnitude makes $\alpha$ comparable across depths). Applied at every denoising step.

**Probe-surprise metric.** Since diffusion models don't expose predictive future representations, they use the trained plausibility probes as a surrogate readout (adapting Garrido et al.'s surprise-as-prediction-error idea). Per inversion step $s$:
$$\psi_s(V) = \text{logit}_{\text{implausible}}(s,V) - \text{logit}_{\text{plausible}}(s,V), \tag{6}$$
aggregated as $\bar{\psi}(V) = \frac{1}{|\mathcal{S}|}\sum_{s} \psi_s(V)$. The surprise shift from intervening on block $b$ relative to an unperturbed baseline from the same noise latent:
$$\Delta_b = \bar{\psi}(V_b) - \bar{\psi}(V_{\text{base}}). \tag{7}$$
Positive $\Delta_b$ means corrupting block $b$ makes the video look less plausible to the probe (the block carries physical signal); near-zero means little signal.

### Derivations / why it works
The approximation-error bound for the explicit reverse step (Appendix A, not in the fetched body) is what justifies treating recovered states as faithful. The empirical validation is the reconstruction round-trip: forward-resampling the recovered noise returns the original video. Section 4.5 provides the key mechanistic evidence — see Ablations.

### Training procedure
No model training — the diffusion backbones are frozen, pretrained, off-the-shelf. Only the linear probes/regressors are trained (60/40 train/test split, held-out accuracy averaged over categories; probes trained independently per block and per inversion step). Reverse sampling uses 100 integration steps for main results.

### Inference / sampling
The whole method *is* an inference-time procedure over frozen generators: reverse-sample (Eq. 4) to recover states, probe, and optionally intervene (Eq. 5) during forward generation.

## Experimental setup

- **Backbones (frozen):** WAN-1.3B, CogVideoX-2B, LTX-2B — three distinct latent-video-diffusion architectural families, all over reconstruction VAEs.
- **Benchmarks:** IntPhys (Riochet et al. 2018) and InfLevel (Weihs et al. 2022), each with plausible/implausible video pairs (violation-of-expectation). Plus Kang et al. 2025's deterministic 2D physics simulator for quantitative variables (initial position $p_0$, velocity $v_0$, mass, size, trajectory type).
- **Baselines:** V-JEPA 2 (ViT-L) and VideoMAE-Large — dedicated self-supervised video encoders — under the same probing protocol; plus probing the raw VAE latents.
- **Metrics:** per-video probe accuracy (classify each clip in isolation — harder than pairwise), pairwise accuracy, linear-regression $R^2$ for physical variables, and probe-surprise shift $\Delta_b$ for interventions.

## Key results

- **Decodability (Figure 2):** all three diffusion models beat both encoders on average; WAN-1.3B reaches **81.27%** per-video vs V-JEPA 2's 71.36%. Pairwise accuracy (Figure 3) also always exceeds V-JEPA/VideoMAE.
- **Not inherited from input:** probing VAE latents directly yields chance (**48–53%**) across all categories and all three VAEs; t-SNE shows plausible/implausible VAE latents are indistinguishable. The signal is *constructed by the diffusion transformer*, not present in its input.
- **Emergence zone (Figure 4):** signal is sustained across nearly the whole reverse trajectory — for every noise level except $t=0$, the best block exceeds 76%; WAN's richest block hits **88.90%**. It is localized in *depth* to the middle third (~blocks 15–25 for WAN and CogVideoX, slightly later for LTX), matching Joseph et al.'s intermediate-depth finding for V-JEPA.
- **Quantitative variables (Figure 6):** linear regressors recover initial position $p_0$ with $R^2 \geq 0.99$ (all models) and initial velocity $v_0$ with $R^2$ between 0.88 (WAN) and ~1.00 (wider DiTs) — matching V-JEPA 2 and VideoMAE-Large. These parameters are never explicitly supervised.

## Ablations

- **Causal localization (Figure 5):** intervening on WAN's 30 blocks shows causal sensitivity is *distributed*, not localized — WAN peaks early (blocks 3–5), CogVideoX shows sharp early dominance, LTX is bimodal (early + late). Critically, the blocks easiest to *decode* from (intermediate depth) are **not** the blocks whose corruption most affects generation (earlier). Interpretation: physical structure is established early, propagated through depth, and becomes most linearly accessible only at intermediate layers.
- **DiT width vs probe accuracy (Figures 6–7):** accuracy *decreases* monotonically with DiT hidden dimension, and does **not** track parameter count — WAN-1.3B is smallest, most compressed, and most decodable. Proposed explanation: a narrow DiT must prioritize coarse spatiotemporal semantics needed for coherent denoising over high-frequency texture, concentrating the physical signal; wider DiTs dilute it across texture-encoding dimensions. (Flagged as suggestive, only 3 architectures.)
- **Physics lives in the flow (Figures 8–9):** the decisive ablation. Reducing reverse-sampling steps from 100 to **20** still reconstructs the video with only mild quality loss, yet probe accuracy **collapses from 0.82 to 0.57** (near chance); at 40 steps accuracy nearly recovers. The discretization needed to *generate* a plausible-looking video is strictly coarser than that needed to *decode* physics. Switching Euler→Heun at 20 steps partially recovers the signal. Conclusion: the physical signal is a property of the *exact continuous ODE trajectory*, not the endpoints — a coarse trajectory is a *different* trajectory.

## Limitations

Authors' own: reverse sampling is approximate and may introduce trajectory errors (fine-grained conclusions may depend on discretization); linear-probe decodability does not imply the model *explicitly represents* physical laws, and the probe-based surprise metric may inherit bias from the learned classifier. More broadly the results are correlational — interventions give partial causal evidence but don't establish mechanistic, human-interpretable encoding of physical laws. Honest-reader flags: the "compression → better probing" trend rests on only three architectures differing in more than width; "outperforms V-JEPA" is under *this* probing protocol and depth-matched setup.

## Why it matters [analyst's view]

The reverse-sampling trick is the reusable contribution: it turns any deterministic-sampler diffusion model into a probeable representation extractor for *real* inputs, not just generated ones — a general interpretability tool, not a physics-specific one. The headline reframes a live debate: the fact that video diffusion models *fail to show* physics in their outputs yet *know* it internally means the generation objective, not the representation, is the bottleneck — which points to physics-aware guidance (steer generation using the internal signal) and to using DiT internal states as world-model latent spaces. The "physics lives in the flow, not the endpoints" result is the most striking and the most falsifiable, and it reframes the intermediate diffusion states as the object of interest rather than the clean sample. Directly complements [[topics/world-models]] work in the vault and Sonia Joseph's interpretability line.

## Open questions / things to verify

- Does the explicit (single-eval) inversion actually stay close to the true implicit trajectory across all three backbones, or does the Euler→Heun recovery signal that Euler is meaningfully off?
- Is the width→dilution story real, or a confound with training recipe/data? A controlled width sweep within one architecture would settle it.
- Can the internal signal actually be *used* — e.g. as guidance to make generations more physical, or as a world-model latent for planning? The conclusion gestures at this but it's untested.
- How does per-video accuracy hold up on harder, real-world (non-simulator) physics beyond IntPhys/InfLevel?

## Connections

- Topic MOCs: [[topics/video-generation]], [[topics/diffusion-models]], [[topics/world-models]], [[topics/jepa]]
- Author indices: [[authors/parsa-esmati]], [[authors/somjit-nath]], [[authors/katja-hofmann]], [[authors/derek-nowrouzezahrai]], [[authors/samira-ebrahimi-kahou]], [[authors/majid-mirmehdi]]
- Related: directly extends Garrido et al. 2025 (intuitive physics from SSL) and [[authors/sonia-joseph]] et al. 2026 "Interpreting physics in video world models" (intermediate-depth emergence) to *generative* diffusion models; contrasts with Kang et al. 2025 (video generation fails physical extrapolation in *outputs*); builds on activation-patching/causal-tracing (Meng et al. 2023, ROME).

## Selected quotes

> "do modern video generation models encode physical knowledge internally, even when their output fails to capture it?" — §1

> "The trajectory is what carries the physics, and a coarse trajectory is a different trajectory." — §4.5

> "the representation is not the training target, but a byproduct of generation." — §4.1
