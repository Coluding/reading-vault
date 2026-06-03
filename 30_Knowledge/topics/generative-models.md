---
type: moc
topic: generative-models
last_updated: 2026-06-03
---

# Generative Models

The line of methods that learn $p(x)$ or $p(x \mid c)$ from samples — covering diffusion / score-matching, latent-variable / variational models, and the broader space of vector-field-defined generative dynamics. The 2026 papers in this vault converge on a shared theme: the *latent geometry* and *dynamics* of generative models matter at least as much as architecture or scale.

## Score-based / diffusion line

- [[papers/yang-2026-replaid-continuous-diffusion]] — first unified scaling comparison between continuous and discrete diffusion language models; closes the AR-vs-continuous-diffusion compute gap from 64× to 20× by protocol alignment; ELBO-variance noise schedule recovers linear cross-entropy "for free."
- [[papers/pao-huang-2026-flux-matching]] — generalizes score matching to the full family of generative vector fields characterized by Fokker–Planck stationarity; turns the vector field into a design choice; enables faster-mixing samplers, interpretable fields, and structured dynamics at 2–4× DSM training cost.
- [[blogs/shing-diffusionblocks]] — *(Sakana AI, ICLR 2026 summary)* casts block-wise network training as a diffusion denoising process so each block trains independently on its own "closeness-to-target" range; ~1/B training memory at performance comparable to end-to-end across ViT/DiT/text models. Diffusion-as-block-roles is another instance of *designing the dynamics a network implements*.
- [[papers/lee-2026-looped-diffusion-lm]] — **LoopMDM**: brings looped (weight-shared) transformers into masked diffusion LMs; looping early-middle layers buys 3.3× training-FLOP efficiency and reasoning gains, with the loop count as an inference-time compute knob. (See also [[topics/looped-transformers]].)

## Efficient inference / sampling

- [[papers/bartosh-2026-dual-rate-diffusion]] — splits the denoiser into a heavy **context encoder** (run every $K$-th step for global structure) and a light **denoising model** (run every step for local detail); 2–4× lower FLOPs at equal/better ImageNet FID, composes with Moment Matching Distillation. Spectral framing: global low-frequencies are slow, so don't recompute them every step.
- [[papers/ding-2024-diffusion-world-model]] — *(also [[topics/world-models]])* diffusion as a one-shot trajectory predictor: generate a whole length-$T$ future in a single pass instead of recursive one-step rollout, killing compounding error in offline RL (44% over one-step models, 4.6× faster than Decision Diffuser).

## Latent-variable / variational

- [[papers/baek-2026-gram]] — GRAM frames recursive reasoning as a latent-variable generative process trained via amortized variational inference; doubles as an unconditional generative model when input conditioning is empty; achieves 99.05% valid Sudoku boards from empty inputs.

## Related topics

- [[topics/diffusion-models]]
- [[topics/score-matching]]
- [[topics/diffusion-language-models]]
- [[topics/variational-inference]]
- [[topics/scaling-laws]]
