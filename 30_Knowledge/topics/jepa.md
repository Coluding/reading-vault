---
type: moc
topic: jepa
last_updated: 2026-05-21
---

# JEPA (Joint Embedding Predictive Architectures)

The line of self-supervised methods, introduced by [LeCun, 2022], that learn to predict the embeddings of future or masked observations in a compact latent space rather than reconstructing pixels. The central engineering challenge is preventing representation collapse — different JEPA recipes choose between EMA + stop-gradient (I-JEPA / V-JEPA), foundation-model encoders frozen for stability (DINO-WM), multi-term VICReg-style losses end-to-end (PLDM), or anti-collapse distributional regularisers (LeWM's SIGReg).

## Foundational

_Not yet in vault — JEPA itself, I-JEPA, V-JEPA, PLDM, DINO-WM remain to be deep-noted._

## Recent

- [[papers/huang-2026-semantic-tube-prediction]] — generalises JEPA to language by enforcing token-trajectory locality on a semantic manifold (the "Geodesic Hypothesis"); claims 16× data-efficiency gain over Chinchilla-style scaling on NL-RX-SYNTH.
- [[papers/maes-2026-leworldmodel]] — first end-to-end stable JEPA from pixels with only a two-term loss (next-embedding MSE + SIGReg); replaces the EMA / stop-gradient / multi-term-loss orthodoxy with a single Cramér–Wold-backed anti-collapse regulariser.

## Related topics

- [[topics/world-models]]
- [[topics/self-predictive-learning]]
- [[topics/representation-collapse]]
- [[topics/scaling-laws]]
