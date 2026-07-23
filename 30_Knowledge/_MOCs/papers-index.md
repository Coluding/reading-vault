---
type: moc
last_updated: 2026-07-23
status: living
---

# Papers Index — all papers, grouped

Every paper in `30_Knowledge/papers/`, exactly **once**, under its *primary* theme — the scannable overview of the whole collection. For multi-membership views (a paper touching several topics), use the topic MOCs in `30_Knowledge/topics/`. Maintained by `/process-inbox`: every new paper note gets one entry here.

**71 papers** as of 2026-07-23.

## World models — core methods & systems (21)

Learned dynamics models: video, latent, and language world models, and what they're used for.

- [[papers/baldassarre-2025-dino-world-models]] (2025) — **DINO-world**: video world model predicting in frozen DINOv2 latent space; beats V-JEPA/COSMOS on dense forecasting at a fraction of the compute.
- [[papers/kerssies-2026-delta-tokens]] (2026) — **DeltaTok/DeltaWorld**: one continuous delta token per frame + Best-of-Many objective; Cosmos-level forecasting at ~2,000× fewer FLOPs.
- [[papers/gao-2025-adaworld]] (2025) — **AdaWorld** (ICML 2025): action-free pretraining via latent actions from unlabeled video; zero-shot action transfer, fast adaptation.
- [[papers/porcher-2026-flowwm]] (2026) — **FlowWM**: stochastic world model via flow matching in frozen-encoder feature space.
- [[papers/maes-2026-leworldmodel]] (2026) — **LeWorldModel**: first end-to-end stable JEPA world model from pixels; two-term loss (MSE + SIGReg), plans up to 48× faster than DINO-WM.
- [[papers/ye-2026-world-action-models]] (2026) — **DreamZero**: 14B World Action Model on a video-diffusion backbone; world action models as zero-shot policies.
- [[papers/ding-2024-diffusion-world-model]] (2024) — **Diffusion World Model**: predicts the whole future trajectory in one pass, killing compounding rollout error; +44% on D4RL offline RL.
- [[papers/jiang-2025-world4rl]] (2025) — **World4RL**: frozen diffusion world model as simulator for RL policy refinement in robotic manipulation.
- [[papers/higuera-2026-visuo-tactile-world-models]] (2026) — multi-task world model fusing vision + tactile; tactile grounding fixes physics-violation failure modes.
- [[papers/zhou-2024-robodreamer]] (2024) — **RoboDreamer** (ICML 2024): compositional text-to-video robot planning via product-of-experts over parsed sub-instructions.
- [[papers/zuo-2026-qwen-agentworld]] (2026) — **Qwen-AgentWorld**: first *native* language world model — next-observation prediction from pre-training onward.
- [[papers/tong-2026-beyond-language-modeling]] (2026) — world-modeling capabilities emerge from generic multimodal pretraining (Transfusion + MoE); vision-vs-language scaling asymmetry.
- [[papers/ding-2024-world-models-survey]] (2024) — *survey*: understand-the-present vs predict-the-future as the field's organizing dichotomy; taxonomy + benchmark catalog.
- [[papers/piriyakulkij-2025-poe-world]] (2025) — **PoE-World** (NeurIPS 2025): world model as an exponentially-weighted product of LLM-synthesized Python expert programs; only method scoring on Montezuma's Revenge from <1000 demo frames.
- [[papers/lei-2024-spartan]] (2024) — **SPARTAN** (NeurIPS 2025): hard attention + sparsity turn a transformer world model's attention into a state-dependent local causal graph; robust to distractors.
- [[papers/shang-2025-roboscape]] (2025) — **RoboScape**: physics-informed embodied world model — joint depth + keypoint consistency losses; near-real synthetic policy data, r=0.953 policy evaluator.
- [[papers/zhu-2026-sana-wm]] (2026) — **SANA-WM** (NVIDIA): minute-scale 720p camera-controlled world modeling on one GPU via hybrid Gated-DeltaNet/sparse-softmax DiT; ~36× throughput.
- [[papers/lu-2026-driftworld]] (2026) — **DriftWorld**: single-forward-pass world model via drifting generative models; 17× faster than diffusion WMs, 0.95+ policy-eval correlation.
- [[papers/mazzaglia-2024-genrl]] (2024) — **GenRL** (NeurIPS 2024): multimodal-foundation world models — task prompts from language/video, policies trained purely in imagination; data-free adaptation.
- [[papers/saxena-2021-clockwork-vae]] (2021) — **Clockwork VAE**: hierarchical latents ticking at exponentially slower clocks; long-horizon video prediction (1000-step digit memory).
- [[papers/ivashkov-2026-sensorimotor-world-models]] (2026) — **SMWM**: inverse dynamics as the sole anti-collapse mechanism in a pixel-trained JEPA world model; latents capture exactly the controllable DoF.

## Physics in video generation — benchmarks & evaluation (7)

Does generated video obey physics, and how do we measure it?

- [[papers/meng-2024-phygenbench]] (2024) — **PhyGenBench**: the seminal physics-commonsense T2V benchmark (160 prompts / 27 laws); best model only 0.51.
- [[papers/gu-2025-phyworldbench]] (2025) — **PhyWorldBench** (ICLR 2026 oral): 1,050 prompts with an anti-physics tier; best model 0.262.
- [[papers/lin-2026-phyground]] (2026) — **PhyGround**: physical-reasoning benchmark for generative world models with open PhyJudge-9B.
- [[papers/begiristain-2026-cronos]] (2026) — **CRONOS**: counterfactual physical consistency under interventions; video generators are alarmingly viewpoint-sensitive.
- [[papers/cao-2026-judgefit]] (2026) — **JudgeFit**: physics VLM-judges disagree with humans and each other; per-VLM taxonomies lift agreement ~32%.
- [[papers/xue-2026-acwm-phys]] (2026) — **ACWM-Phys**: action-conditioned world models across 4 physics categories; OoD failure tracks task complexity, not physics.
- [[papers/yao-2026-apple-pi]] (2026) — **Apple-π**: law-grounded physical-*reasoning* benchmark (Perception→Formulation→Deduction chain-of-frames); best video model 0.473.

## Physics in video generation — methods & analysis (7)

Making video models physics-faithful, and probing whether they already are.

- [[papers/zhao-2026-phyworld]] (2026) — **PhyWorld**: physics-faithful world model for video generation (DPO-based physics alignment).
- [[papers/xiong-2026-physalign]] (2026) — **PhysAlign**: LoRA adapter aligning Wan2.2 I2V to a V-JEPA2 teacher + 3D depth; better physics at no quality cost.
- [[papers/yuan-2026-physics-alignment]] (2026) — inference-time physics alignment of video generators with latent world models (FAIR).
- [[papers/esmati-2026-invisible-hand-physics]] (2026) — video diffusion models internally *encode* physics even when outputs violate it.
- [[papers/joseph-2026-physics-video-world-models]] (2026) — interpretability: physics is encoded distributedly with a "Physics Emergence Zone" mid-network.
- [[papers/chen-2026-actionable-simulators]] (2026) — *position survey*: "visual conflation" — realism ≠ dynamics; world models must become actionable simulators with closed-loop evaluation.
- [[papers/wang-2025-wisa]] (2025) — **WISA**: physics-aware T2V via decomposed physical conditions (29 categories, Mixture-of-Physical-Experts attention) on WISA-32K.

## Latent spaces for generation — RAE / tokenizer line (4)

Frozen vision-foundation-model features as the latent space for diffusion.

- [[papers/zheng-2025-rae-dit]] (2025) — **the RAE paper**: frozen DINOv2/SigLIP2/MAE encoder + trained ViT decoder replaces SD-VAE; 1.51 FID w/o guidance, ~47× training speedup.
- [[papers/liu-2026-improving-rae-reconstruction]] (2026) — **LV-RAE**: residual low-level encoder + noise-robust decoder fixes RAE reconstruction and off-manifold hypersensitivity.
- [[papers/liu-2026-geometric-autoencoder]] (2026) — **GAE**: bottleneck teacher alignment + hyperspherical RMSNorm latent + dynamic noise; gFID 1.31 w/o CFG at d=32.
- [[papers/li-2026-semantic-autoencoder]] (2026) — **S-AE** (CVPR 2026): one latent space for understanding *and* generation off frozen DINOv3.

## Diffusion, flow matching & generative methods (5)

Core generative-modeling machinery: objectives, couplings, efficiency.

- [[papers/malnick-2026-designing-ot-flows]] (2026) — OT-optimal prior *by design* (low-frequency projections); straighter flows with no OT solver.
- [[papers/pao-huang-2026-flux-matching]] (2026) — **Flux Matching**: generalizes score matching to all Fokker–Planck-consistent vector fields.
- [[papers/cai-2026-mode-mean-seeking]] (2026) — decoupled mean-seeking + mode-seeking heads for fast minute-scale video generation.
- [[papers/bartosh-2026-dual-rate-diffusion]] (2026) — heavy context encoder every K-th step + light denoiser every step; 2–4× cheaper inference.
- [[papers/yang-2026-replaid-continuous-diffusion]] (2026) — closes the AR-vs-continuous-diffusion compute gap from 64× to 20×; ELBO-variance schedule = linear cross-entropy.

## Looped transformers & latent reasoning (7)

Recurrence-in-depth: weight-shared loops, fixed points, and reasoning in hidden space.

- [[papers/wang-2025-hierarchical-reasoning-model]] (2025) — **HRM**: two recurrent modules at slow/fast timescales; 27M params solve Sudoku-Extreme / ARC-AGI without pretraining.
- [[papers/baek-2026-gram]] (2026) — **GRAM**: recursive reasoning as latent-variable generation via amortized variational inference; width-based inference-time scaling.
- [[papers/movahedi-2026-fixed-point-reasoners]] (2026) — **FPRM**: stable deep looping via fixed-point formulation.
- [[papers/lee-2026-looped-diffusion-lm]] (2026) — **LoopMDM**: looped transformers meet masked diffusion LMs; 3.3× training-FLOP efficiency.
- [[papers/deng-2026-lt2-looped]] (2026) — **LT2**: subquadratic mixers in looped transformers; "looping turns compute into context."
- [[papers/chen-2026-training-free-looped]] (2026) — loops a frozen checkpoint's mid-block at inference via the ODE view; no training.
- [[papers/zou-2026-latent-thought-flow]] (2026) — **LTF**: LLM reasoning inside continuous hidden space instead of decoded chain-of-thought.

## Self-predictive & representation learning (5)

Learning representations by predicting your own future embeddings.

- [[papers/guo-2022-byol-explore]] (2022) — **BYOL-Explore**: BYOL latent-prediction loss as joint world model + exploration bonus.
- [[papers/khetarpal-2024-byol-ac]] (2024) — ODE theory unifying action-conditional BYOL variants.
- [[papers/daithankar-2026-temporal-difference-vision]] (2026) — **TDV**: image representations from video via temporal differences, minus strong inductive biases.
- [[papers/huang-2026-semantic-tube-prediction]] (2026) — generalizes JEPA to language via the "Geodesic Hypothesis"; claims 16× data-efficiency over Chinchilla.
- [[papers/groger-2026-aristotelian-view]] (2026) — challenges the Platonic Representation Hypothesis with an Aristotelian view.

## Robotics & embodied learning (4)

Robot learning beyond the world-model papers above.

- [[papers/jiang-2024-dexmimicgen]] (2024) — **DexMimicGen**: a handful of teleop demos → large bimanual dexterous imitation datasets.
- [[papers/karcini-2026-robots-beyond-vla]] (2026) — *position*: generalist robotics is bottlenecked by grounding, not policy scaling; four missing pillars.
- [[papers/shi-2026-gpc-motor-control]] (2026) — **GPC**: LLM-recipe generative pretraining for transferable motor control.
- [[papers/jiang-2026-robottt]] (2026) — **RoboTTT**: TTT layers give a VLA policy 8K-timestep context at constant cost; 79% vs 42% on real bimanual assembly; context as a scaling axis.

## Exploration & open-endedness (6)

The older lineage: novelty, self-invented problems, and exploration as the bottleneck.

- [[papers/jiang-2022-rethinking-exploration]] (2022) — *position*: general intelligence requires rethinking exploration; outer/inner-loop framework.
- [[papers/clune-2019-ai-gas]] (2019) — **AI-GAs**: three pillars of AI-generating algorithms; pre-LLM "environment is the bottleneck" thesis.
- [[papers/schmidhuber-2013-powerplay]] (2013) — **PowerPlay**: self-invention of training problems via simplest-unsolvable-problem search.
- [[papers/lehman-2011-novelty-search]] (2011) — abandoning objectives: selecting for behavioural novelty beats objective-based search on deceptive problems.
- [[papers/momennejad-2026-compositional-open-ended]] (2026) — *theory*: a compositional formal object for open-ended intelligence.
- [[papers/zhang-2026-learnable-novelty]] (2026) — **Learnable novelty** (Zhang & Levin): one differentiable objective unifying complexity, abstraction, exploration; stable intrinsic reward, 9/10 envs.

## Architectures, optimization & reliability (5)

Everything about the network itself: width allocation, attention, optimizers, faults.

- [[papers/bayat-2026-tapered-language-models]] (2026) — per-layer MLP width allocation instead of uniform width.
- [[papers/litman-2026-attention-priors]] (2026) — attention as entropic OT with an implicit uniform prior; principled learnable priors incl. attention-sink bias.
- [[papers/knight-2026-riemannian-low-rank]] (2026) — Riemannian GD over ten geometries for low-rank params; honest negative result vs AdamW.
- [[papers/kumar-2026-supervised-memory-training]] (2026) — **SMT**: pretraining nonlinear RNNs without BPTT.
- [[papers/galil-2025-sign-bit-flips]] (2025) — **Deep Neural Lesion**: data-free attack destroying DNNs by flipping a few weight sign bits.
