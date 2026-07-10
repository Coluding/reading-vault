---
type: moc
topic: world-models
last_updated: 2026-07-10
---

# World Models

Learned models of environment dynamics — pixel-, latent-, or token-level predictors used for planning, exploration, or as substrates for downstream reasoning.

## Foundational
- [[papers/guo-2022-byol-explore]] — learns world representation + dynamics jointly via a BYOL latent-prediction loss; the prediction error doubles as the exploration bonus.

## Recent
- [[papers/esmati-2026-invisible-hand-physics]] — This paper asks whether video diffusion models internally *encode* physical structure even when their generated outputs violate physics
- [[papers/ye-2026-world-action-models]] — The paper introduces **DreamZero**, a 14B **World Action Model (WAM)** built on a pretrained image-to-video diffusion backbone (Wan2.1-I2V-14B) that j
- [[papers/porcher-2026-flowwm]] — **FlowWM** is a stochastic visual world model that runs **flow matching directly inside the high-dimensional feature space of a frozen pretrained enco
- [[papers/jiang-2025-world4rl]] — **World4RL** is a two-stage framework that uses a **diffusion-based world model as a high-fidelity, frozen simulator** in which to refine an imitation
- [[papers/yuan-2026-physics-alignment]] — The paper (from FAIR / Meta Superintelligence Labs, dated March 2, 2026) shows that a large share of the physics-implausibility in state-of-the-art vi
- [[blogs/interlatent-ai-robotics]] — A first-principles tour of how modern "physical AI" robot policies are built, framed around the idea that a robot policy is just a function mapping ob
- [[papers/joseph-2026-physics-video-world-models]] — interpretability study showing modern video world models encode physics in a distributed, non-factorised manner with a sharp "Physics Emergence Zone" mid-network.
- [[papers/higuera-2026-visuo-tactile-world-models]] — multi-task world model that fuses vision with tactile sensing; claims tactile grounding fixes physics-violation failure modes of vision-only models and yields 33% better object permanence in rollouts.
- [[papers/maes-2026-leworldmodel]] — first end-to-end stable JEPA world model from pixels; two-term loss (prediction MSE + SIGReg), 15M params, plans up to 48× faster than DINO-WM, latent encodes recoverable physical structure.
- [[papers/tong-2026-beyond-language-modeling]] — argues world-modeling capabilities emerge from generic unified multimodal pretraining (Transfusion + MoE); identifies a vision-vs-language scaling asymmetry.
- [[papers/ding-2024-diffusion-world-model]] — diffusion world model that predicts a whole future trajectory of states+rewards in **one pass** (conditioned on state, action, return-to-go), sidestepping the compounding error of recursive one-step rollouts; +44% over one-step models on D4RL offline RL, closes the model-based-vs-model-free gap. (Also [[topics/generative-models]].)

## Perspectives & surveys
- [[blogs/mccormick-world-models]] — General Intuition manifesto/survey: World Models as a new foundation-model class ($P(s_{t+1}\mid s_t,a_t)$ vs video's $P(x_{t+1}\mid x_t)$); maps the field's four waves and every major lab (latent/JEPA vs generative/diffusion, VLAs vs World-Model agents); bets on action-labeled gaming data.
- [[blogs/joseph-world-models-interpretability]] — reframes "world model" as a foundation model's *internal* causal structure (the Internal World Model); argues world models and interpretability are the same program (causal discovery + white-box evaluation). Explicit contrast to the external/generative view above.

## Organizations
- [[websites/embodyx]] — physical-AI startup shipping edge-first VLM/VLA foundation models; founders (Weiwei Chen, Yanzhi Wang) authored the vault's physics-faithful world-model work ([[papers/zhao-2026-phyworld]], [[papers/lin-2026-phyground]]).

## Related topics
- [[topics/jepa]]
- [[topics/self-predictive-learning]]
- [[topics/video-world-models]]
- [[topics/interpretability]]
- [[topics/robotics]]
- [[topics/tactile-sensing]]
- [[topics/multimodal-pretraining]]
