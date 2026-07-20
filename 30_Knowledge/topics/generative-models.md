---
type: moc
topic: generative-models
last_updated: 2026-07-20
---

# Generative Models

The line of methods that learn $p(x)$ or $p(x \mid c)$ from samples — covering diffusion / score-matching, latent-variable / variational models, and the broader space of vector-field-defined generative dynamics. The 2026 papers in this vault converge on a shared theme: the *latent geometry* and *dynamics* of generative models matter at least as much as architecture or scale.

## Score-based / diffusion line

- [[papers/yang-2026-replaid-continuous-diffusion]] — first unified scaling comparison between continuous and discrete diffusion language models; closes the AR-vs-continuous-diffusion compute gap from 64× to 20× by protocol alignment; ELBO-variance noise schedule recovers linear cross-entropy "for free."
- [[papers/pao-huang-2026-flux-matching]] — generalizes score matching to the full family of generative vector fields characterized by Fokker–Planck stationarity; turns the vector field into a design choice; enables faster-mixing samplers, interpretable fields, and structured dynamics at 2–4× DSM training cost.
- [[blogs/shing-diffusionblocks]] — *(Sakana AI, ICLR 2026 summary)* casts block-wise network training as a diffusion denoising process so each block trains independently on its own "closeness-to-target" range; ~1/B training memory at performance comparable to end-to-end across ViT/DiT/text models. Diffusion-as-block-roles is another instance of *designing the dynamics a network implements*.
- [[papers/lee-2026-looped-diffusion-lm]] — **LoopMDM**: brings looped (weight-shared) transformers into masked diffusion LMs; looping early-middle layers buys 3.3× training-FLOP efficiency and reasoning gains, with the loop count as an inference-time compute knob. (See also [[topics/looped-transformers]].)

## Flow matching / optimal-transport coupling

- [[papers/malnick-2026-designing-ot-flows]] — stop *solving* OT; **design a prior whose identity coupling to the data is OT-optimal** (low-frequency image projections). Straightens flow-matching trajectories (>2× lower curvature) with no OT solver, composing with latent diffusion, CFG, and one-step MeanFlow. (See [[topics/flow-matching]], [[topics/optimal-transport]].)
- [[papers/cai-2026-mode-mean-seeking]] — decoupled diffusion transformer: a mean-seeking flow-matching head (long-video coherence from scarce data) + a mode-seeking distribution-matching head distilled from a frozen short-video teacher; FM head discarded at inference for fast minute-scale video.
- [[blogs/dieleman-diffusion-integral]] — unifying tutorial on **flow maps** and the three consistency rules (compositionality / Lagrangian / Eulerian); the map of the 2024–26 few-step-sampling literature.
- [[blogs/flow-based-llms-intro]] — flow-based language models: softmax + cross-entropy = the Variational-Flow-Matching objective on the simplex; flows are distillable into Categorical Flow Maps (discrete diffusion is not).
- [[blogs/jiha-autoregression-vs-diffusion]] — AR vs diffusion as two parameterizations of one optimal-transport problem (Knothe–Rosenblatt rearrangement vs learned Brenier map).
- [[blogs/accelerated-diffusion-tutorial]] — CVPR 2026 "FastGen" tutorial: faster sampling, efficient samplers, and distillation for real-time image/video generation.

## Efficient inference / sampling

- [[papers/bartosh-2026-dual-rate-diffusion]] — splits the denoiser into a heavy **context encoder** (run every $K$-th step for global structure) and a light **denoising model** (run every step for local detail); 2–4× lower FLOPs at equal/better ImageNet FID, composes with Moment Matching Distillation. Spectral framing: global low-frequencies are slow, so don't recompute them every step.
- [[papers/ding-2024-diffusion-world-model]] — *(also [[topics/world-models]])* diffusion as a one-shot trajectory predictor: generate a whole length-$T$ future in a single pass instead of recursive one-step rollout, killing compounding error in offline RL (44% over one-step models, 4.6× faster than Decision Diffuser).

## Latent spaces for generation (RAE / tokenizer line)

- [[papers/zheng-2025-rae-dit]] — **the RAE paper** (Xie lab): frozen DINOv2/SigLIP2/MAE encoder + trained ViT decoder as the latent space for DiTs; rank argument (Theorem 1) for width ≥ token dim; 1.51 FID w/o guidance on ImageNet-256, argues RAE should replace SD-VAE as the default.
- [[papers/liu-2026-improving-rae-reconstruction]] — **LV-RAE**: shallow residual encoder adds back the low-level detail frozen VFM features drop (PSNR 32.32 near-lossless); diagnoses + fixes decoder hypersensitivity to off-manifold latents via noise-augmented fine-tuning.
- [[papers/liu-2026-geometric-autoencoder]] — **GAE**: three principled fixes for VFM-semantic compact latents — bottleneck-level teacher alignment, hyperspherical RMSNorm (no KL), dynamic noise sampling; gFID 1.31 w/o CFG.
- [[papers/li-2026-semantic-autoencoder]] — **S-AE** (CVPR 2026): "unified latent space" — semantically discriminative *and* reconstruction-faithful off frozen DINOv3, via high channel dim + semantic regularization; strong on low-data/high-res domains.
- [[papers/kerssies-2026-delta-tokens]] — **DeltaTok**: temporal cousin of the RAE line — a frozen-VFM-feature *delta* per frame pair compressed to one continuous token turns video into a 1-D sequence for world modeling.

## Latent-variable / variational

- [[papers/baek-2026-gram]] — GRAM frames recursive reasoning as a latent-variable generative process trained via amortized variational inference; doubles as an unconditional generative model when input conditioning is empty; achieves 99.05% valid Sudoku boards from empty inputs.

## Recent
- [[papers/xiong-2026-physalign]] — **PhysAlign**: post-trains a Wan2.2 I2V flow-matching model into a **physically coherent** generator via a LoRA adapter — Gram-based relational alignment to a frozen V-JEPA2 teacher + explicit 3D depth constraints, trained on ~3K synthetic rigid-body clips; wins all VBench-I2V metrics with *no* quality tradeoff. (Also [[topics/world-models]], [[topics/video-generation]].)
- [[papers/porcher-2026-flowwm]] — **FlowWM** is a stochastic visual world model that runs **flow matching directly inside the high-dimensional feature space of a frozen pretrained enco
- [[papers/shi-2026-gpc-motor-control]] — GPC (Generative Pretrained Controllers) builds general-purpose, reusable controllers for physics-based character animation by borrowing the LLM recipe
- [[blogs/mccormick-world-models]] — A long (~18k word) manifesto from Packy McCormick, co-written with the team at his World Model startup **General Intuition**, arguing that World Model
- [[papers/ding-2024-world-models-survey]] — survey whose "future prediction" branch catalogs the generative world-model landscape (Sora, Cosmos, Genie, GAIA-1) alongside model-based-RL and LLM world knowledge.

## Related topics

- [[topics/diffusion-models]]
- [[topics/flow-matching]]
- [[topics/optimal-transport]]
- [[topics/distillation]]
- [[topics/score-matching]]
- [[topics/diffusion-language-models]]
- [[topics/variational-inference]]
- [[topics/scaling-laws]]
