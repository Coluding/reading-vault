---
type: moc
topic: recurrence
last_updated: 2026-07-23
---

# Recurrence

_Vault notes touching recurrence._

## Notes
- [[papers/movahedi-2026-fixed-point-reasoners]] — FPRM (Fixed-Point Reasoning Model) is a non-hierarchical Looped Transformer for latent reasoning that (a) makes deep looping trainable by switching fr
- [[papers/kumar-2026-supervised-memory-training]] — The paper introduces **Supervised Memory Training (SMT)**, a way to train nonlinear RNNs *without* backpropagation through time (BPTT)
- [[papers/jiang-2026-robottt]] — **RoboTTT**: TTT layers turn GR00T N1.7 into a recurrent policy with gradient-updated fast weights — 8K-timestep context at constant cost; 79% vs 42% on real bimanual assembly; one-shot imitation from one human video; context length as a new scaling axis.
- [[papers/saxena-2021-clockwork-vae]] — **Clockwork VAE**: hierarchical latent video model, each level ticking exponentially slower — slow content migrates up for free (KL paid only at active ticks); pure-latent rollouts; Minecraft accurate 400+ frames, MNIST digit identity kept 1000 steps (baselines ~300).
