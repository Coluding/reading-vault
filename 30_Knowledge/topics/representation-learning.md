---
type: moc
topic: representation-learning
last_updated: 2026-07-23
---

# Representation Learning

_Vault notes touching representation learning._

## Notes
- [[papers/daithankar-2026-temporal-difference-vision]] — TDV (Temporal Difference in Vision) is a self-supervised recipe for learning image representations from video that deliberately drops the strong induc
- [[papers/groger-2026-aristotelian-view]] — This paper argues that the **Platonic Representation Hypothesis (PRH)** — the claim that neural nets trained on different data/modalities converge to
- [[blogs/joseph-world-models-interpretability]] — Sonia Joseph proposes a fourth definition of "world model" that cuts against the dominant generative/3D/latent-prediction framings: the **Internal Wor
- [[papers/zheng-2025-rae-dit]] — **the RAE paper**: frozen representation encoder (DINOv2/SigLIP2/MAE) + trained ViT decoder replaces SD-VAE for latent diffusion; overturns "semantic encoders can't reconstruct" (rFID 0.16 with MAE-B) — the anchor of the vault's RAE cluster.
- [[papers/liu-2026-improving-rae-reconstruction]] — **LV-RAE**: freezes the VFM "base manifold," trains a shallow residual encoder for the low-level detail VFM features drop; near-lossless reconstruction (PSNR 32.32, CKNNA 0.987) at f16d768.
- [[papers/liu-2026-geometric-autoencoder]] — **GAE**: bottleneck-level alignment to a downsampled DINOv2 teacher + hyperspherical RMSNorm latent (no KL) + dynamic noise sampling; strong linear probing (69.4% at d=32) alongside gFID 1.31.
- [[papers/li-2026-semantic-autoencoder]] — **S-AE** (CVPR 2026): reconstructs images from *frozen* DINOv3 features via ViT decoder + semantic regularization loss; one latent space serving both understanding and generation.
- [[papers/baldassarre-2025-dino-world-models]] — **DINO-world**: frozen DINOv2 patch features as the *state space* of a video world model — representation quality directly bounds forecasting quality.
- [[papers/kerssies-2026-delta-tokens]] — **DeltaTok**: frame-to-frame change in frozen DINOv3 feature space compressed to a single continuous token; the "no change = previous frame" prior comes free from the delta representation.
- [[papers/zhang-2026-learnable-novelty]] — **Learnable novelty** (Zhang & Levin): intelligence as maximizing the epiplexity a bounded observer can compress into a model; closed-form reservoir estimator ranks rule 110 top of all 88 ECA; MNIST probe 0.53→0.89 unsupervised; stable intrinsic RL reward (9/10 envs, zero collapses).
- [[papers/mazzaglia-2024-genrl]] — **GenRL** (NeurIPS 2024): frozen InternVideo2 video–language space aligned to a vision-only world-model latent via a learned connector; language/video-prompted tasks trained purely in imagination; 0.80 vs 0.70 best baseline over 35 reward-free tasks; data-free adaptation variant.
- [[papers/ivashkov-2026-sensorimotor-world-models]] — **SMWM**: JEPA-style world model whose sole anti-collapse mechanism is a single-step inverse-dynamics head ($\mathcal{L}_{fwd} + \lambda\mathcal{L}_{inv}$); latent rank matches controllable DoF, distractors filtered out; 84% vs 59% (SIGReg) on OGBench-Cube.
