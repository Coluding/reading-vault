---
type: moc
topic: world-models
last_updated: 2026-05-21
---

# World Models

Learned models of environment dynamics — pixel-, latent-, or token-level predictors used for planning, exploration, or as substrates for downstream reasoning.

## Foundational
- [[papers/guo-2022-byol-explore]] — learns world representation + dynamics jointly via a BYOL latent-prediction loss; the prediction error doubles as the exploration bonus.

## Recent
- [[papers/joseph-2026-physics-video-world-models]] — interpretability study showing modern video world models encode physics in a distributed, non-factorised manner with a sharp "Physics Emergence Zone" mid-network.
- [[papers/higuera-2026-visuo-tactile-world-models]] — multi-task world model that fuses vision with tactile sensing; claims tactile grounding fixes physics-violation failure modes of vision-only models and yields 33% better object permanence in rollouts.
- [[papers/maes-2026-leworldmodel]] — first end-to-end stable JEPA world model from pixels; two-term loss (prediction MSE + SIGReg), 15M params, plans up to 48× faster than DINO-WM, latent encodes recoverable physical structure.
- [[papers/tong-2026-beyond-language-modeling]] — argues world-modeling capabilities emerge from generic unified multimodal pretraining (Transfusion + MoE); identifies a vision-vs-language scaling asymmetry.

## Related topics
- [[topics/jepa]]
- [[topics/self-predictive-learning]]
- [[topics/video-world-models]]
- [[topics/interpretability]]
- [[topics/robotics]]
- [[topics/tactile-sensing]]
- [[topics/multimodal-pretraining]]
