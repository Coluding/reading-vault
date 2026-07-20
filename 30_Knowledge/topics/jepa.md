---
type: moc
topic: jepa
last_updated: 2026-07-20
---

# JEPA (Joint Embedding Predictive Architectures)

The line of self-supervised methods, introduced by [LeCun, 2022], that learn to predict the embeddings of future or masked observations in a compact latent space rather than reconstructing pixels. The central engineering challenge is preventing representation collapse — different JEPA recipes choose between EMA + stop-gradient (I-JEPA / V-JEPA), foundation-model encoders frozen for stability (DINO-WM), multi-term VICReg-style losses end-to-end (PLDM), or anti-collapse distributional regularisers (LeWM's SIGReg).

## Foundational

_Not yet in vault — JEPA itself, I-JEPA, V-JEPA, PLDM, DINO-WM remain to be deep-noted._

## Recent
- [[papers/esmati-2026-invisible-hand-physics]] — This paper asks whether video diffusion models internally *encode* physical structure even when their generated outputs violate physics
- [[papers/daithankar-2026-temporal-difference-vision]] — TDV (Temporal Difference in Vision) is a self-supervised recipe for learning image representations from video that deliberately drops the strong induc
- [[blogs/mccormick-world-models]] — A long (~18k word) manifesto from Packy McCormick, co-written with the team at his World Model startup **General Intuition**, arguing that World Model

- [[papers/huang-2026-semantic-tube-prediction]] — generalises JEPA to language by enforcing token-trajectory locality on a semantic manifold (the "Geodesic Hypothesis"); claims 16× data-efficiency gain over Chinchilla-style scaling on NL-RX-SYNTH.
- [[papers/maes-2026-leworldmodel]] — first end-to-end stable JEPA from pixels with only a two-term loss (next-embedding MSE + SIGReg); replaces the EMA / stop-gradient / multi-term-loss orthodoxy with a single Cramér–Wold-backed anti-collapse regulariser.
- [[papers/baldassarre-2025-dino-world-models]] — **DINO-world** (LeCun co-author): sidesteps the collapse problem entirely by *freezing* the DINOv2 encoder and training only the latent predictor — the "foundation-model-encoder frozen for stability" recipe at 1.1B-predictor scale, beating V-JEPA on dense forecasting.

## Related topics

- [[topics/world-models]]
- [[topics/self-predictive-learning]]
- [[topics/predictive-processing]]
- [[topics/representation-collapse]]
- [[topics/scaling-laws]]
