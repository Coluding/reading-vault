---
type: moc
topic: self-supervised-learning
last_updated: 2026-07-23
---

# Self Supervised Learning

_Vault notes touching self supervised learning._

## Notes
- [[papers/daithankar-2026-temporal-difference-vision]] — TDV (Temporal Difference in Vision) is a self-supervised recipe for learning image representations from video that deliberately drops the strong induc
- [[papers/maes-2026-leworldmodel]] — 
- [[papers/baldassarre-2025-dino-world-models]] — **DINO-world**: next-frame teacher-forcing in frozen DINOv2 latent space on ~60M uncurated web videos — self-supervised video world modeling built on a self-supervised image backbone.
- [[papers/gao-2025-adaworld]] — **AdaWorld**: latent actions extracted from raw video with *no action labels* (β-VAE information bottleneck on consecutive-frame pairs) — self-supervised action discovery for world-model pretraining.
- [[papers/li-2026-semantic-autoencoder]] — **S-AE**: builds its unified generation/understanding latent on frozen self-supervised DINOv3 features.
- [[papers/liu-2026-geometric-autoencoder]] — **GAE**: distills self-supervised DINOv2 semantics into a compact hyperspherical diffusion latent via a learned Semantic Teacher.
- [[papers/zheng-2025-rae-dit]] — **RAE**: frozen self-supervised encoders (DINOv2, MAE) turn out to be excellent *generative* latent spaces once decoder and DiT width are handled correctly.
- [[papers/ivashkov-2026-sensorimotor-world-models]] — **SMWM**: JEPA-style world model whose sole anti-collapse mechanism is a single-step inverse-dynamics head ($\mathcal{L}_{fwd} + \lambda\mathcal{L}_{inv}$); latent rank matches controllable DoF, distractors filtered out; 84% vs 59% (SIGReg) on OGBench-Cube.
