---
type: moc
topic: self-predictive-learning
last_updated: 2026-07-08
---

# Self-Predictive Learning

Latent-bootstrapping objectives for representation learning, especially in RL. The agent predicts future latent representations of its own encoder (BYOL-style) instead of pixels or rewards, so the loss has no auxiliary terms.

## Foundational
- [[papers/guo-2022-byol-explore]] — BYOL applied as both representation loss and curiosity bonus; superhuman on hard Atari exploration.

## Recent
- [[papers/daithankar-2026-temporal-difference-vision]] — TDV (Temporal Difference in Vision) is a self-supervised recipe for learning image representations from video that deliberately drops the strong induc
- [[papers/khetarpal-2024-byol-ac]] — ODE-based theory for the action-conditional case (BYOL-AC) plus a variance-derived variant (BYOL-VAR); empirically beats fixed-policy BYOL-Π.

## Related topics
- [[topics/representation-learning-rl]]
- [[topics/world-models]]
- [[topics/predictive-processing]]
