---
type: moc
topic: world-models
last_updated: 2026-07-22
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
- [[papers/begiristain-2026-cronos]] — intervention-based benchmark for **counterfactual physical consistency**: holds a physical event fixed while varying viewpoint / scene / appearance / object-category; finds open-source video generators are most fragile to **viewpoint** (predictions are strongly view-dependent, not 3D-aware) and scaling Cosmos 2B→14B *hurt* success.
- [[papers/meng-2024-phygenbench]] — the **seminal** physics-commonsense T2V benchmark (160 prompts / 27 physical laws / 4 domains) + hierarchical VLM+LLM evaluator **PhyGenEval**; best model (Gen-3) scores only 0.51, and scaling & prompt-engineering are both insufficient. The reference PhyWorld/PhyGround react to.
- [[papers/gu-2025-phyworldbench]] — 1,050-prompt T2V physics benchmark (ICLR 2026 oral) with a novel **anti-physics** tier and an MLLM-judge (CAP) score; best model (Pika 2.0) reaches only 0.262 joint success. The benchmark [[papers/lin-2026-phyground]] critiques for its closed-source judge.
- [[papers/cao-2026-judgefit]] — VLM physics-judges disagree with humans *and each other*; **JudgeFit** discovers a **per-VLM** error taxonomy via human-calibrated pruning, +~32% Spearman agreement across 16 VLMs. Directly targets the judge-bias problem behind PhyGround's PhyJudge-9B.
- [[papers/xiong-2026-physalign]] — LoRA adapter makes Wan2.2 I2V **physically coherent** via synthetic rigid-body clips + Gram-relational alignment to a frozen V-JEPA2 teacher + depth constraints; better physics *and* visual quality, free at inference. Method sibling to PhyWorld (alignment vs DPO).
- [[papers/xue-2026-acwm-phys]] — benchmark for **action-conditioned** video world models across 4 physics categories (rigid/deformable/particle/kinematics); OoD generalization tracks *task complexity*, not physics category — evidence models lean on appearance over physics.
- [[papers/baldassarre-2025-dino-world-models]] — **DINO-world**: generalist video world model predicting future frames in *frozen DINOv2 latent space* (~1.1B cross-attention predictor, 60M web videos, smooth-L1 + 3-axial RoPE); beats V-JEPA/COSMOS on dense forecasting (+6.3 mIoU VSPW @0.5s) at a tiny fraction of COSMOS's compute; post-trainable with lightweight action blocks for planning.
- [[papers/kerssies-2026-delta-tokens]] — **DeltaTok/DeltaWorld**: compresses each frame-to-frame *change* in frozen DINOv3 feature space into a single continuous delta token (~1,024× token reduction) + Best-of-Many objective (K noise queries, backprop through the closest); best-of-20 beats Cosmos on dense forecasting with 35× fewer params and ~2,000× fewer FLOPs.
- [[papers/gao-2025-adaworld]] — **AdaWorld** (ICML 2025): action-free world-model pretraining — a β-VAE bottleneck extracts continuous latent actions from consecutive-frame pairs; SVD-based autoregressive diffusion WM over ~2B frames / 1000+ envs; zero-shot action transfer (FVD 767 vs 1545 on LIBERO) and fast adaptation (56.7% Procgen goal-reaching vs 26% action-agnostic).
- [[papers/zhou-2024-robodreamer]] — **RoboDreamer** (ICML 2024): compositional text-to-video robot planning — parse instruction into action/relation sub-phrases, average per-factor diffusion scores (product-of-experts); zero-shot recombination lifts unseen-task success to 81.3% vs 46.9% (AVDC) on RT-1.
- [[papers/zuo-2026-qwen-agentworld]] — **Qwen-AgentWorld**: first *native* language world model — a text generator trained from pre-training onward to predict the next environment observation given interaction history and action; positions LWMs as simulators for general (tool-use/computer) agents.

## Perspectives & surveys
- [[papers/ding-2024-world-models-survey]] — Tsinghua survey organizing the field around *understand-the-present vs predict-the-future* (Ha–Schmidhuber 2018 vs LeCun's JEPA); two-branch taxonomy (implicit representation vs future prediction), four application domains, six open problems; the field's best current **map** with a benchmarks catalog.
- [[papers/chen-2026-actionable-simulators]] — Oxford position survey naming **"visual conflation"** (visual realism ≠ dynamics understanding); argues world models must become *actionable simulators* — causal structure, domain constraints, long-horizon stability — with closed-loop decision-oriented evaluation; distinguishes perceptual vs *dynamical/causal* hallucinations, medical decision-making as epistemic stress test.
- [[blogs/mccormick-world-models]] — General Intuition manifesto/survey: World Models as a new foundation-model class ($P(s_{t+1}\mid s_t,a_t)$ vs video's $P(x_{t+1}\mid x_t)$); maps the field's four waves and every major lab (latent/JEPA vs generative/diffusion, VLAs vs World-Model agents); bets on action-labeled gaming data.
- [[blogs/joseph-world-models-interpretability]] — reframes "world model" as a foundation model's *internal* causal structure (the Internal World Model); argues world models and interpretability are the same program (causal discovery + white-box evaluation). Explicit contrast to the external/generative view above.

## Organizations
- [[websites/embodyx]] — physical-AI startup shipping edge-first VLM/VLA foundation models; founders (Weiwei Chen, Yanzhi Wang) authored the vault's physics-faithful world-model work ([[papers/zhao-2026-phyworld]], [[papers/lin-2026-phyground]]).

## Syntheses
- [[_synthesis/physics-grounding-video-world-models]] — physics grounding in video/world models: benchmarks, alignment methods, internal probing (12 papers, 2026-07-22)

## Related topics
- [[topics/jepa]]
- [[topics/self-predictive-learning]]
- [[topics/video-world-models]]
- [[topics/interpretability]]
- [[topics/robotics]]
- [[topics/tactile-sensing]]
- [[topics/multimodal-pretraining]]
