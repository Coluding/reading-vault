---
type: moc
topic: generative-models
last_updated: 2026-05-27
---

# Generative Models

The line of methods that learn $p(x)$ or $p(x \mid c)$ from samples — covering diffusion / score-matching, latent-variable / variational models, and the broader space of vector-field-defined generative dynamics. The 2026 papers in this vault converge on a shared theme: the *latent geometry* and *dynamics* of generative models matter at least as much as architecture or scale.

## Score-based / diffusion line

- [[papers/yang-2026-replaid-continuous-diffusion]] — first unified scaling comparison between continuous and discrete diffusion language models; closes the AR-vs-continuous-diffusion compute gap from 64× to 20× by protocol alignment; ELBO-variance noise schedule recovers linear cross-entropy "for free."
- [[papers/pao-huang-2026-flux-matching]] — generalizes score matching to the full family of generative vector fields characterized by Fokker–Planck stationarity; turns the vector field into a design choice; enables faster-mixing samplers, interpretable fields, and structured dynamics at 2–4× DSM training cost.
- [[blogs/shing-diffusionblocks]] — *(Sakana AI, ICLR 2026 summary)* casts block-wise network training as a diffusion denoising process so each block trains independently on its own "closeness-to-target" range; ~1/B training memory at performance comparable to end-to-end across ViT/DiT/text models. Diffusion-as-block-roles is another instance of *designing the dynamics a network implements*.

## Latent-variable / variational

- [[papers/baek-2026-gram]] — GRAM frames recursive reasoning as a latent-variable generative process trained via amortized variational inference; doubles as an unconditional generative model when input conditioning is empty; achieves 99.05% valid Sudoku boards from empty inputs.

## Related topics

- [[topics/diffusion-models]]
- [[topics/score-matching]]
- [[topics/diffusion-language-models]]
- [[topics/variational-inference]]
- [[topics/scaling-laws]]
