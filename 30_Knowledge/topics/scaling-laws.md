---
type: moc
topic: scaling-laws
last_updated: 2026-07-08
---

# Scaling Laws

Empirical power-law characterizations of how model loss / capability scales with compute, data, and parameters. Three papers in vault deal with scaling-law deltas: one revising the data term of Chinchilla, one revising the AR-vs-continuous-diffusion compute gap, one identifying a vision-vs-language asymmetry inside unified multimodal training. Together they suggest the headline scaling-law numbers from any era are *protocol-bound*, not laws of nature.

## Recent
- [[papers/bayat-2026-tapered-language-models]] — Transformers (and their modern relatives) allocate the same MLP width to every layer
- [[blogs/haegele-magnitude-direction]] — The post introduces **Magnitude-Direction Decoupling (MD)**: constrain each weight matrix's *direction* to a fixed-norm sphere while giving it a separ

- [[papers/tong-2026-beyond-language-modeling]] — IsoFLOP analysis on Transfusion-style native multimodal pretraining; "vision is significantly more data-hungry than language"; MoE architecturally compensates for the asymmetry.
- [[papers/huang-2026-semantic-tube-prediction]] — geometric prior (Geodesic Hypothesis) + JEPA-style trajectory regulariser; claims 16× LLM data-efficiency over Chinchilla on NL-RX-SYNTH.
- [[papers/yang-2026-replaid-continuous-diffusion]] — first unified scaling comparison between continuous and discrete diffusion LMs; closes the AR-vs-continuous gap from 64× to 20× via protocol alignment; ELBO-variance minimisation recovers a linear cross-entropy noise schedule.
- [[blogs/lampinen-continual-learning-problems]] — argues **scale is the quiet solver** of continual learning: catastrophic interference and loss of plasticity are largely artifacts of small models / hard task boundaries, and larger pretrained models forget less (sparser, more orthogonal gradients). Reframes the real open problem as *positive transfer / cumulative learning*, not forgetting.

## Related topics

- [[topics/jepa]]
- [[topics/multimodal-pretraining]]
- [[topics/diffusion-language-models]]
- [[topics/data-efficiency]]
