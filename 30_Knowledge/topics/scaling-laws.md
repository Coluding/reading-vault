---
type: moc
topic: scaling-laws
last_updated: 2026-05-22
---

# Scaling Laws

Empirical power-law characterizations of how model loss / capability scales with compute, data, and parameters. Three papers in vault deal with scaling-law deltas: one revising the data term of Chinchilla, one revising the AR-vs-continuous-diffusion compute gap, one identifying a vision-vs-language asymmetry inside unified multimodal training. Together they suggest the headline scaling-law numbers from any era are *protocol-bound*, not laws of nature.

## Recent

- [[papers/tong-2026-beyond-language-modeling]] — IsoFLOP analysis on Transfusion-style native multimodal pretraining; "vision is significantly more data-hungry than language"; MoE architecturally compensates for the asymmetry.
- [[papers/huang-2026-semantic-tube-prediction]] — geometric prior (Geodesic Hypothesis) + JEPA-style trajectory regulariser; claims 16× LLM data-efficiency over Chinchilla on NL-RX-SYNTH.
- [[papers/yang-2026-replaid-continuous-diffusion]] — first unified scaling comparison between continuous and discrete diffusion LMs; closes the AR-vs-continuous gap from 64× to 20× via protocol alignment; ELBO-variance minimisation recovers a linear cross-entropy noise schedule.

## Related topics

- [[topics/jepa]]
- [[topics/multimodal-pretraining]]
- [[topics/diffusion-language-models]]
- [[topics/data-efficiency]]
