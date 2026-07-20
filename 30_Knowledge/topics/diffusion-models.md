---
type: moc
topic: diffusion-models
last_updated: 2026-07-20
---

# Diffusion Models

Generative models that learn to reverse a noising process — equivalently, to
integrate a (deterministic) velocity/score field connecting noise to data. The
vault's 2026 diffusion notes are dominated by **inference acceleration** (fewer
ODE steps via straighter trajectories, distillation, or split denoisers) and by
**reframings** that connect diffusion to flow matching and optimal transport.

## Tutorials / surveys
- [[blogs/dieleman-diffusion-integral]] — diffusion sampling as integration of a velocity field; unifies few-step methods via the **flow-map** abstraction and three consistency rules.
- [[blogs/accelerated-diffusion-tutorial]] — CVPR 2026 "FastGen" tutorial: speeding up sampling, training efficient samplers, and distillation for real-time image/video generation.
- [[blogs/jiha-autoregression-vs-diffusion]] — AR vs diffusion as two parameterizations of one optimal-transport problem.

## Methods
- [[papers/malnick-2026-designing-ot-flows]] — OT-optimal prior-by-design (low-frequency projections) for straighter, few-step flow/diffusion trajectories.
- [[papers/pao-huang-2026-flux-matching]] — generalizes score matching to the full family of Fokker–Planck-consistent generative vector fields; the vector field becomes a design choice.
- [[blogs/shing-diffusionblocks]] — casts block-wise network training as diffusion denoising so each block trains independently; ~1/B training memory.

## Recent
- [[papers/esmati-2026-invisible-hand-physics]] — This paper asks whether video diffusion models internally *encode* physical structure even when their generated outputs violate physics
- [[papers/jiang-2025-world4rl]] — **World4RL** is a two-stage framework that uses a **diffusion-based world model as a high-fidelity, frozen simulator** in which to refine an imitation
- [[papers/yuan-2026-physics-alignment]] — The paper (from FAIR / Meta Superintelligence Labs, dated March 2, 2026) shows that a large share of the physics-implausibility in state-of-the-art vi
- [[blogs/mccormick-world-models]] — A long (~18k word) manifesto from Packy McCormick, co-written with the team at his World Model startup **General Intuition**, arguing that World Model
- [[papers/zhou-2024-robodreamer]] — **RoboDreamer**: product-of-experts over per-sub-instruction diffusion score functions (scores averaged at sampling) for compositional zero-shot robot video planning.

## Latent spaces / tokenizers (RAE line)
- [[papers/zheng-2025-rae-dit]] — **RAE**: frozen representation encoder + trained ViT decoder replaces SD-VAE; DiT width ≥ token dim (Theorem 1) + dimension-dependent noise-schedule shift make high-dim latent diffusion work — FID 1.51 w/o guidance / 1.13 with, ~47× training speedup over SiT-XL.
- [[papers/liu-2026-improving-rae-reconstruction]] — **LV-RAE**: residual low-level encoder on a frozen VFM manifold + noise-robust decoder fine-tuning ($\sigma\sim\mathcal{U}(0,0.2)$ + GAN loss) fixes RAE's reconstruction and off-manifold hypersensitivity; gFID 2.42 w/o guidance.
- [[papers/liu-2026-geometric-autoencoder]] — **GAE**: latent alignment to a downsampled DINOv2 teacher, hyperspherical RMSNorm instead of KL, dynamic noise sampling (σ-VAE); gFID 1.31 at 800 epochs w/o CFG on ImageNet-256 with a 32-dim latent.
- [[papers/li-2026-semantic-autoencoder]] — **S-AE**: high-channel-dim autoencoder off frozen DINOv3 features with a semantic regularization loss; DiTs on S-AE latents converge faster/more stably than on DC-AE latents.

## Related topics
- [[topics/flow-matching]]
- [[topics/optimal-transport]]
- [[topics/distillation]]
- [[topics/generative-models]] — broader generative-modeling MOC (incl. discrete/continuous diffusion-LM scaling, dual-rate diffusion, diffusion world models)
