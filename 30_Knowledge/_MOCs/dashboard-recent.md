---
type: moc
last_updated: 2026-07-27
status: living

---

# Dashboard — Recent

Most recent items added to the vault, regardless of category. Refreshed by
`/process-inbox`.

## This week

_(Triaged 2026-07-27 — cleared 07-26 + 07-27 raw days: **1 unique paper** from 4 bullets (Worker double-posted both items). A hard turn away from the world-models wave into **LLM inference systems** — the first speculative-decoding paper in the vault. It activates the long-dangling `topics/llm-inference` MOC, which [[topics/gpu-optimization]] had been linking to since the KV-cache blog landed in May.)_

### Papers
- [[papers/cheng-2026-dspark]] — **DSpark** (DeepSeek-AI + PKU): semi-autoregressive speculative decoding — deep parallel backbone plus a rank-256 Markov transition head — with confidence-scheduled verification sized against a profiled engine-capacity curve. +30.9% accepted length over Eagle3 (Qwen3-4B); 60–85% faster per-user generation in live DeepSeek-V4 serving. Notable ablation: a 2-layer DSpark beats a 5-layer DFlash.

### Rejected
- Readwise & Reader Changelog, "July 24, 2026" (rw-id `01kyg6ws1mdbnkhgt5vact8etr`) — off-topic: product release notes for the capture tooling itself.

## Earlier

_(Triaged 2026-07-23 — cleared 07-22 + 07-23 raw days: **12 unique papers** from 20 bullets (Worker double-posted most). Two clusters: **structured/causal world models** (PoE-World, SPARTAN, GenRL, Clockwork VAE — several were `_needs note_` gaps flagged by the actionable-simulators survey) and a continuation of the **physics-grounding wave** (RoboScape, WISA, Apple-π). Both title-less arXiv bullets identified from content: 2503.08153 = WISA, 2406.18043 = GenRL — each closes a gap flagged in [[_synthesis/physics-grounding-video-world-models]].)_

### Papers
- [[papers/piriyakulkij-2025-poe-world]] — **PoE-World** (NeurIPS 2025): world model as a product of hundreds of LLM-synthesized Python expert programs; only method to score on Montezuma's Revenge from <1000 demo frames.
- [[papers/lei-2024-spartan]] — **SPARTAN** (NeurIPS 2025): hard attention + sparsity regularisation make attention a state-dependent local causal graph; SHD 1.17 vs 6.29, distractor-robust.
- [[papers/shang-2025-roboscape]] — **RoboScape**: physics-informed embodied world model via joint depth + keypoint-consistency losses; r=0.953 as a policy evaluator.
- [[papers/wang-2025-wisa]] — **WISA**: physics-aware T2V by decomposing physics into textual/qualitative/quantitative conditions (29 categories, Mixture-of-Physical-Experts attention) on WISA-32K.
- [[papers/zhu-2026-sana-wm]] — **SANA-WM** (NVIDIA): 60 s of 720p camera-controlled world modeling on one GPU (hybrid Gated DeltaNet / sparse-softmax DiT, ~36× throughput).
- [[papers/yao-2026-apple-pi]] — **Apple-π**: law-grounded physical-*reasoning* benchmark (Perception→Formulation→Deduction chain-of-frames); best video model 0.473; unified models fail Deduction (~0.40).
- [[papers/zhang-2026-learnable-novelty]] — **Learnable novelty** (Zhang & Levin): one differentiable objective unifying complexity, abstraction, and exploration; rule 110 ranked top of all 88 ECA unsupervised; stable intrinsic RL reward.
- [[papers/jiang-2026-robottt]] — **RoboTTT**: TTT fast-weight layers give a VLA policy 8K-timestep context at constant cost; 79% vs 42% on real bimanual assembly; one-shot imitation from a single human video.
- [[papers/lu-2026-driftworld]] — **DriftWorld**: single-pass world model via drifting generative models (no iterative denoising); 17× faster than diffusion WMs; 0.95+ policy-eval correlation.
- [[papers/mazzaglia-2024-genrl]] — **GenRL** (NeurIPS 2024): multimodal-foundation world models — language/video task prompts, policies trained purely in imagination; data-free adaptation.
- [[papers/saxena-2021-clockwork-vae]] — **Clockwork VAE** (2021 classic): hierarchical latents at exponentially slower clocks; 1000-step digit memory where baselines forget by ~300.
- [[papers/ivashkov-2026-sensorimotor-world-models]] — **SMWM** (Schölkopf lab): inverse dynamics as the *sole* anti-collapse mechanism in a pixel-trained JEPA world model; latents capture exactly the controllable DoF.

### Duplicates
- Worker double-posted 8 of the 12 bullets (both days); deduplicated during routing.

### Rejected
- _none this pass_

## Earlier — 07-20 backlog pass

_(Triaged 2026-07-20 — cleared the 07-07 → 07-16 raw backlog: the 07-10 world-models + RAE-latent-space wave, plus two blogs. Deep notes were written in an earlier interrupted session; this pass completed MOC links, author indices, routing, and the commit. 16 unique items from 31 bullets — the Worker double-posted most. Two clusters dominate: **VFM-latent-space world models** (DINO-world, DeltaTok, AdaWorld) and the **representation-autoencoder line** (RAE, LV-RAE, GAE, S-AE).)_

### Papers
- [[papers/zheng-2025-rae-dit]] — **the RAE paper** (Xie lab): frozen DINOv2/SigLIP2/MAE encoder + trained ViT decoder replaces SD-VAE for DiTs; width ≥ token dim (Theorem 1); 1.51 FID w/o guidance on ImageNet-256, ~47× training speedup. Anchor of the RAE cluster.
- [[papers/liu-2026-improving-rae-reconstruction]] — **LV-RAE**: shallow residual encoder on a frozen VFM manifold restores the low-level detail RAEs drop (PSNR 32.32 near-lossless); noise-robust decoder fine-tuning fixes off-manifold hypersensitivity.
- [[papers/liu-2026-geometric-autoencoder]] — **GAE**: bottleneck alignment to a downsampled DINOv2 teacher + hyperspherical RMSNorm latent (no KL) + dynamic noise sampling; gFID 1.31 w/o CFG at d=32.
- [[papers/li-2026-semantic-autoencoder]] — **S-AE** (CVPR 2026): one latent space for understanding *and* generation off frozen DINOv3; strong on low-data/high-res domains; faster DiT convergence than DC-AE.
- [[papers/baldassarre-2025-dino-world-models]] — **DINO-world** (Meta, LeCun co-author): video world model predicting in frozen DINOv2 latent space; beats V-JEPA/COSMOS on dense forecasting at a tiny fraction of the compute; post-trainable action blocks for planning.
- [[papers/kerssies-2026-delta-tokens]] — **DeltaTok/DeltaWorld**: one continuous delta token per frame (change in frozen DINOv3 space) + Best-of-Many objective; Cosmos-level forecasting at 35× fewer params, ~2,000× fewer FLOPs.
- [[papers/gao-2025-adaworld]] — **AdaWorld** (ICML 2025): action-free pretraining via continuous latent actions (β-VAE bottleneck over frame pairs); zero-shot action transfer, fast adaptation to new action spaces.
- [[papers/zhou-2024-robodreamer]] — **RoboDreamer** (ICML 2024): compositional text-to-video robot planning — product-of-experts over parsed sub-instructions; 81.3% unseen-task success on RT-1.
- [[papers/zuo-2026-qwen-agentworld]] — **Qwen-AgentWorld**: first *native* language world model — next-observation prediction from pre-training onward; an LLM as environment simulator for general agents.
- [[papers/ding-2024-world-models-survey]] — Tsinghua survey: understand-the-present vs predict-the-future as the field's organizing dichotomy; the best current map + benchmark catalog.
- [[papers/chen-2026-actionable-simulators]] — Oxford position survey: "visual conflation" — realism ≠ dynamics; world models must become *actionable simulators* with closed-loop decision-oriented evaluation.
- _(also routed this pass, deep-noted earlier via /paper-search: [[papers/zhao-2026-phyworld]], [[papers/lin-2026-phyground]], and [[papers/kerssies-2026-delta-tokens]]'s siblings below.)_

### Blogs
- [[blogs/velickovic-gdl-book]] — **GDL textbook** (Veličković, MIT Press 2026) released chapter-by-chapter as free drafts; tracking note — triggered creation of [[topics/geometric-deep-learning]].
- [[blogs/readwise-changelog-2026-07-10]] — Reader changelog: Markdown uploads, Daily Review API endpoint, `language` save parameter, MCP response slimming.

### Duplicates
- "World Models: Computing the Uncomputable" (07-07 bullet) — same rw-id already processed in the 07-08 backfill → [[blogs/mccormick-world-models]].

### Rejected
- Readwise & Reader Changelog "July 17, 2026" (07-20 bullet, ×2 double-post) — `reject: off-topic` (product changelog, not AI/math/CS). _Note: departs from prior practice of low-priority changelog notes; payload JSON retained if reversal is wanted._

_(2026-07-10 — **/paper-search**: 6 physics-in-video-generation siblings to [[papers/zhao-2026-phyworld]] & [[papers/lin-2026-phyground]], pulled directly from arXiv (not via Readwise — `rw_id: ""`, `read_state: skimmed`). Split benchmark axis / method axis.)_

### Papers (paper-search — physics-in-video cluster)
- [[papers/meng-2024-phygenbench]] — **PhyGenBench** (ICML 2025): the *seminal* physics-commonsense T2V benchmark — 160 prompts / 27 laws / 4 domains + hierarchical **PhyGenEval** judge; best model (Gen-3) only 0.51; scaling & prompt-engineering both insufficient.
- [[papers/gu-2025-phyworldbench]] — **PhyWorldBench** (ICLR 2026 oral): 1,050-prompt benchmark with an **anti-physics** tier + MLLM-judge (CAP); best model (Pika 2.0) only 0.262 joint success. *The benchmark PhyGround critiques for its closed-source judge.*
- [[papers/cao-2026-judgefit]] — **JudgeFit**: physics VLM-judges disagree with humans and each other; a per-VLM discovered taxonomy lifts human agreement ~32% across 16 VLMs.
- [[papers/begiristain-2026-cronos]] — **CRONOS**: intervention-based counterfactual-consistency benchmark (hold the physical event fixed, vary viewpoint/scene/appearance); video generators are alarmingly **viewpoint**-sensitive — pattern-matching, not 3D physics.
- [[papers/xiong-2026-physalign]] — **PhysAlign** *(method sibling to PhyWorld)*: a LoRA adapter aligning Wan2.2 I2V features to a V-JEPA2 teacher + 3D depth for physics coherence, with no visual-quality tradeoff; free at inference.
- [[papers/xue-2026-acwm-phys]] — **ACWM-Phys**: benchmarks **action-conditioned** video world models across rigid/deformable/particle/kinematics; OoD failure tracks task complexity, not physics category. (Bridges to robotics.)

_(Triaged 2026-07-08 — **backfill** of the 2026-06-15 → 07-07 Worker-outage gap: world models, video/physics diffusion, reasoning & looped transformers, representation learning, robotics. 14 papers, 7 blogs, 1 reject.)_

### Papers
- [[papers/zou-2026-latent-thought-flow]] — **Latent Thought Flow: Efficient Latent Reasoning in Large Language Models**: Latent Thought Flow (LTF) makes LLM reasoning happen inside the model's continuous hidden space instead of decoded chain-of-thought tokens, and — unli.
- [[papers/esmati-2026-invisible-hand-physics]] — **The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show**: This paper asks whether video diffusion models internally *encode* physical structure even when their generated outputs violate physics.
- [[papers/daithankar-2026-temporal-difference-vision]] — **You Don't Need Strong Assumptions: Visual Representation Learning via Temporal Differences**: TDV (Temporal Difference in Vision) is a self-supervised recipe for learning image representations from video that deliberately drops the strong induc.
- [[papers/jiang-2024-dexmimicgen]] — **DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning**: DexMimicGen (DexMG) is an automated data-generation system that turns a *handful* of human teleoperation demos into a large imitation-learning dataset.
- [[papers/groger-2026-aristotelian-view]] — **Revisiting the Platonic Representation Hypothesis: An Aristotelian View**: This paper argues that the **Platonic Representation Hypothesis (PRH)** — the claim that neural nets trained on different data/modalities converge to.
- [[papers/movahedi-2026-fixed-point-reasoners]] — **Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers**: FPRM (Fixed-Point Reasoning Model) is a non-hierarchical Looped Transformer for latent reasoning that (a) makes deep looping trainable by switching fr.
- [[papers/ye-2026-world-action-models]] — **World Action Models are Zero-shot Policies**: The paper introduces **DreamZero**, a 14B **World Action Model (WAM)** built on a pretrained image-to-video diffusion backbone (Wan2.1-I2V-14B) that j.
- [[papers/porcher-2026-flowwm]] — **Flow Matching in Feature Space for Stochastic World Modeling**: **FlowWM** is a stochastic visual world model that runs **flow matching directly inside the high-dimensional feature space of a frozen pretrained enco.
- [[papers/kumar-2026-supervised-memory-training]] — **Pretraining Recurrent Networks without Recurrence**: The paper introduces **Supervised Memory Training (SMT)**, a way to train nonlinear RNNs *without* backpropagation through time (BPTT).
- [[papers/yuan-2026-physics-alignment]] — **Inference-time Physics Alignment of Video Generative Models with Latent World Models**: The paper (from FAIR / Meta Superintelligence Labs, dated March 2, 2026) shows that a large share of the physics-implausibility in state-of-the-art vi.
- [[papers/bayat-2026-tapered-language-models]] — Tapered Language Models: Transformers (and their modern relatives) allocate the same MLP width to every layer.
- [[papers/momennejad-2026-compositional-open-ended]] — A Compositional Framework for Open-ended Intelligence: This is a **theoretical / position paper** (no experiments) that proposes a formal object for open-ended intelligence built from a finite set of primi.
- [[papers/jiang-2025-world4rl]] — World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation: **World4RL** is a two-stage framework that uses a **diffusion-based world model as a high-fidelity, frozen simulator** in which to refine an imitation.
- [[papers/shi-2026-gpc-motor-control]] — GPC: Large-Scale Generative Pretraining for Transferable Motor Control: GPC (Generative Pretrained Controllers) builds general-purpose, reusable controllers for physics-based character animation by borrowing the LLM recipe.

### Blogs
- [[blogs/joseph-world-models-interpretability]] — **World models and interpretability are two sides of the same coin**: Sonia Joseph proposes a fourth definition of "world model" that cuts against the dominant generative/3D/latent-prediction framings: the **Internal Wor.
- [[blogs/interlatent-ai-robotics]] — **An Overview of Modern AI Robotics from First Principles**: A first-principles tour of how modern "physical AI" robot policies are built, framed around the idea that a robot policy is just a function mapping ob.
- [[blogs/mccormick-world-models]] — **World Models: Computing the Uncomputable**: A long (~18k word) manifesto from Packy McCormick, co-written with the team at his World Model startup **General Intuition**, arguing that World Model.
- [[blogs/haegele-magnitude-direction]] — Improving Neural Network Training by Decoupling the Magnitude and Direction of Weight Vectors: The post introduces **Magnitude-Direction Decoupling (MD)**: constrain each weight matrix's *direction* to a fixed-norm sphere while giving it a separ.
- _Readwise changelogs (low priority):_ [[blogs/readwise-changelog-2026-06-26]], [[blogs/readwise-changelog-2026-07-03]], [[blogs/readwise-changelog-2026-06-19]].

### Websites
- [[websites/embodyx]] — *(ingested 07-10)* **EmbodyX**: Silicon Valley physical-AI startup building edge-first VLM/VLA foundation models for robots, vehicles, and cameras. Founders Weiwei Chen & Yanzhi Wang (Northeastern) authored the vault's [[papers/zhao-2026-phyworld]] and [[papers/lin-2026-phyground]]; backed by NSF, NVIDIA, Google AI for Startups.

### Rejected
- Loom (loom.com) — off-topic product page (not AI/math/CS).

_(Triaged 2026-06-22 — the flow-matching / OT / generative cluster, plus robotics, attention, reliability, and geometry. Source-saved 06-10 to 06-15.)_

### Papers
- [[papers/malnick-2026-designing-ot-flows]] — *(06-10)* **Optimal Transport Flow Matching by Design**: stop *solving* OT — design a prior (low-frequency image projections) whose identity coupling to the data is empirically OT-optimal; >2× straighter trajectories and better few-step FID (+20% vs OT-FM, +38% vs IFM at 1 step on FFHQ) with no flow-model changes; composes with MeanFlow.
- [[papers/cai-2026-mode-mean-seeking]] — *(06-10)* Decoupled Diffusion Transformer for **fast long video**: a mean-seeking flow-matching head (coherence from scarce long-video data) + a mode-seeking distribution-matching head distilled from a frozen short-video teacher; FM head discarded at inference for ~30 s minute-scale video.
- [[papers/karcini-2026-robots-beyond-vla]] — *(06-10)* Position paper: generalist robotics is mis-framed as policy scaling; the real bottleneck is **grounding** abundant unstructured behavioural data. Four missing pillars — physical data engine, task-preserving retargeting, physics-grounded world models, self-improving deployment loops. VLAs are one layer, not the stack.
- [[papers/litman-2026-attention-priors]] — *(06-10)* Recasts attention as **entropic optimal transport** with an implicit uniform prior; positional encodings are heuristic approximations of an EOT-derived $\log\pi$ term; introduces a principled learnable prior incl. a dedicated key-only attention-sink bias.
- [[papers/galil-2025-sign-bit-flips]] — *(06-10)* **Deep Neural Lesion**: data-free, optimization-free attack that destroys DNNs by flipping a few weight *sign bits*; works across vision, detection, and reasoning LLMs. (reliability/hardware-faults cluster.)

### Blogs
- [[blogs/dieleman-diffusion-integral]] — *(06-10)* Sander Dieleman's unifying tutorial on **flow maps** (predict any point on a noise→data path from any other) via three consistency rules (compositionality / Lagrangian / Eulerian); the map of the 2024–26 few-step-sampling literature.
- [[blogs/jiha-autoregression-vs-diffusion]] — *(06-10)* AR vs diffusion as two solutions to **one optimal-transport problem**: AR = Knothe–Rosenblatt rearrangement (triangular, exact, ordering-biased); diffusion/flow = learned unconstrained Brenier map.
- [[blogs/flow-based-llms-intro]] — *(06-10)* Floor Eijkelboom's intro to **flow-based language models**: softmax + cross-entropy is exactly the Variational-Flow-Matching objective on the simplex; flows distill into one/two-step Categorical Flow Maps (discrete diffusion can't).
- [[blogs/accelerated-diffusion-tutorial]] — *(06-11)* CVPR 2026 "FastGen" tutorial: faster sampling, training efficient samplers, and distillation for real-time image/video generation.
- [[blogs/nicolas-riemannian-manifolds]] — *(06-15)* Foundations of Riemannian geometry for Geometric Deep Learning (manifolds, tangent spaces, geodesics, exp/log maps, metric, connection, curvature, Lie groups); part 1 of a series, payoff is computing on data manifolds with `Geomstats`.

## Last week

### Papers
- [[papers/ding-2024-diffusion-world-model]] — *(06-03)* Diffusion World Model: predicts a whole future trajectory in one diffusion pass (conditioned on state/action/return-to-go), killing the compounding error of recursive one-step rollouts; +44% over one-step models on D4RL, 4.6× faster than Decision Diffuser, closes the model-based-vs-model-free gap.
- [[papers/lee-2026-looped-diffusion-lm]] — *(06-03, looped-models cluster)* **LoopMDM**: first looped transformer for masked diffusion LMs; selectively loops early-middle layers for up to 3.3× fewer training FLOPs, +8.5 GSM8K, with loop count as an inference-time compute knob.
- [[papers/deng-2026-lt2-looped]] — *(06-03, looped-models cluster)* **LT2**: replaces softmax attention in looped transformers with subquadratic (linear/sparse) mixers — "looping turns compute into context" (rank-$T$ memory / $O(Tw)$ receptive field); hybrid beats the full-attention loop at ~5× decode throughput; converts pretrained Ouro → Ouro-Hybrid-1.4B.
- [[papers/chen-2026-training-free-looped]] — *(06-03, looped-models cluster)* loops a frozen checkpoint's mid-block at inference (no training) via the forward-Euler/ODE view; Runge–Kutta integration, +2.64 pp MMLU-Pro on Qwen3-4B; layer-mode needed for MoE.
- [[papers/bartosh-2026-dual-rate-diffusion]] — *(05-31)* heavy context encoder (every $K$-th step) + light denoiser (every step) → 2–4× cheaper diffusion inference at equal/better ImageNet FID; composes with Moment Matching Distillation (student beats teacher).
- [[papers/knight-2026-riemannian-low-rank]] — *(06-03)* Riemannian gradient descent over ten geometries for rank-factored attention params; honest **negative result** — does not conclusively beat AdamW at small scale; nice framing of AdamW/Muon as steepest descent under $\ell_\infty$/spectral norms.

### Blogs
- [[blogs/clark-surfing-uncertainty-precis]] — *(06-04, predictive-processing cluster)* Andy Clark's own précis of *Surfing Uncertainty*: the brain as a hierarchical prediction machine; prediction-as-bootstrap of world knowledge; action as active inference. The cognitive-science taproot of self-supervised prediction.
- [[blogs/alexander-surfing-uncertainty-review]] — *(06-04, predictive-processing cluster)* Scott Alexander's SSC explainer: two precision-weighted streams reconciled by Bayes at every layer; "controlled hallucination," surprisal minimization, autism (over-precise priors) vs schizophrenia (weak priors). The ML-legible on-ramp.
- [[blogs/buskell-surfing-uncertainty-review]] — *(06-04, predictive-processing cluster)* Buskell's BSPS scholarly review: PP as a "mid-level organizational sketch"; flags an ambiguity in what "moving up the hierarchy" means (scope vs gist vs hyperprior). The honest counterweight.
- [[blogs/lampinen-continual-learning-problems]] — *(06-03, Andrew Lampinen / Infinite Faculty)* argues scale + pretraining largely dissolve catastrophic interference & loss of plasticity; the real frontier of continual learning is positive transfer / cumulative learning, and dissolving the in-context-vs-parametric boundary.

## Older

### Late May – early June (exploration / open-endedness, JEPA & world-models, GPU cluster)

#### Papers
- [[papers/clune-2019-ai-gas]] — *(manual citation chase from Jiang 2022)* "AI-Generating Algorithms" with three pillars: meta-learning architectures, meta-learning learning algorithms, generating environments. Pre-LLM articulation of the "data/environment is the bottleneck" thesis.
- [[papers/schmidhuber-2013-powerplay]] — *(manual citation chase from Jiang 2022)* algorithmic framework for self-invention of training problems via Kolmogorov-complexity-ordered search over (new-task, solver-modification) pairs.
- [[papers/lehman-2011-novelty-search]] — *(manual citation chase from Jiang 2022)* foundational novelty-search paper: abandoning explicit objectives and selecting purely for behavioural novelty can outperform objective-based search on deceptive problems.
- [[papers/jiang-2022-rethinking-exploration]] — position paper (Meta/UCL/Cohere, late 2022): exploration is the bottleneck for general intelligence; proposes the outer/inner-loop generalized-exploration framework unifying SL and RL.
- [[papers/pao-huang-2026-flux-matching]] — generalizes score matching to the full Fokker–Planck-consistent family of generative vector fields; turns the vector field into a design choice; faster mixing + interpretable fields + structured dynamics at 2–4× DSM training cost.
- [[papers/wang-2025-hierarchical-reasoning-model]] — foundational HRM (Sapient Intelligence): two recurrent modules at slow/fast timescales, DEQ-style 1-step gradient; 27M params, 1000 examples solve Sudoku-Extreme / Maze-Hard / ARC-AGI without pretraining or CoT.
- [[papers/baek-2026-gram]] — extends HRM into probabilistic multi-trajectory recursion via stochastic latent transitions + amortized variational inference; introduces width-based inference-time scaling; 99.7% N-Queens 8×8 vs HRM 78.7%.
- [[papers/yang-2026-replaid-continuous-diffusion]] — closes the AR-vs-continuous-diffusion compute gap from 64× to 20× via protocol alignment; SOTA 22.1 PPL on OpenWebText among continuous DLMs; shows ELBO-variance noise schedule = linear cross-entropy.
- [[papers/guo-2022-byol-explore]] — BYOL-style self-predictive loss as joint world-model + intrinsic reward; solves DM-HARD-8 without demos.
- [[papers/khetarpal-2024-byol-ac]] — ODE theory for action-conditional BYOL representations; unifies BYOL-Π / BYOL-AC / BYOL-VAR.
- [[papers/joseph-2026-physics-video-world-models]] — interpretability of physics in video encoders; identifies the Physics Emergence Zone and a circular population code for motion direction.
- [[papers/higuera-2026-visuo-tactile-world-models]] — multi-task world model fusing vision + tactile; claims tactile grounding fixes physics-violation failure modes and yields up to 35% higher zero-shot real-robot success.
- [[papers/tong-2026-beyond-language-modeling]] — controlled from-scratch native-multimodal pretraining (Transfusion + MoE); finds a scaling asymmetry (vision more data-hungry than language) and world-modeling capabilities emerge from generic training.
- [[papers/huang-2026-semantic-tube-prediction]] — generalises JEPA to language via the "Geodesic Hypothesis"; claims 16× LLM data-efficiency over Chinchilla on NL-RX-SYNTH.
- [[papers/maes-2026-leworldmodel]] — first stable end-to-end JEPA from pixels; two-term loss (prediction MSE + SIGReg) replaces EMA / stop-gradient / multi-term-loss orthodoxy; plans up to 48× faster than DINO-WM.

#### Blogs
- [[blogs/shing-diffusionblocks]] — *(Sakana AI, 05-27, ICLR 2026 summary)* train networks one block at a time by casting block-wise dynamics as diffusion denoising; ~1/B training memory at performance comparable to end-to-end across 5 architectures (ViT/DiT/Masked-Diffusion/AR/Recurrent-depth). Bonus: replaces K-step BPTT with a single forward pass for looped Transformers. _Headline numbers in the paper, not yet verified._
- [[blogs/blainsmith-just-use-go]] — *(05-27, off the AI/math axis — CS/backend polemic)* "use boring Go": stdlib-as-framework, cheap goroutines, single static binary deploy. Low durable value; kept for the audit trail + as a boring-monolith reference.
- [[blogs/bansal-kv-cache]] — *(GPU/inference cluster, 05-25)* KV-cache management as the binding constraint in LLM serving; six-era evolution (contiguous → PagedAttention → heterogeneous → distributed → unified) mirroring OS memory management. Llama-3-70B at 8K × 32 reqs = 81.9 GB of KV cache alone.
- [[blogs/bierling-coalesced-matmul]] — *(05-25, author = vault owner)* naive vs coalesced CUDA matmul; a one-line thread-index remap → coalesced global loads → ~10.5× (1433 ms → 136 ms on an A100). Stays memory-bound; tiling next.
- [[blogs/bierling-prefix-scan]] — *(05-25, author = vault owner)* parallel prefix scan: shared-memory Hillis–Steele ($O(\log N)$ depth) + divide-and-conquer recursion over block sums for arbitrary sizes.
- [[blogs/pytorch-torch-amp]] — *(05-25)* `torch.amp` reference: `autocast` (per-op dtype) + `GradScaler` (fp16 loss scaling); wrap forward only, cast back to fp32 on exit, bf16-pretrained models overflow in fp16.
- [[blogs/aiwithmaha-cuda-concepts]] — *(05-25)* beginner glossary of 20 CUDA concepts (kernel/grid/block/thread/warp, global vs shared memory, occupancy, streams, tensor cores, cuDNN). Shared-vocabulary entry point for the GPU cluster.
- [[blogs/biswas-rlm-deep-dive]] — practitioner deep-dive on Recursive Language Models: pass-by-reference agentic scaffold (LLM operates inside a Python REPL with `llm_query` for subagents); 60M-character Lex Fridman transcript in 4 minutes for $0.20.

---

## How this dashboard works

This file is rewritten at the end of every `/process-inbox` run. It shows:

- **This week**: items added in the last 7 days, grouped by category.
- **Last week**: items added 7–14 days ago.
- **Older**: rolled-up by month.

For deeper queries see the other dashboards in `30_Knowledge/_MOCs/`:

- `dashboard-deep-read.md` — items marked `priority: high` and `read_state: queued`
- `dashboard-orphans.md` — notes with no inbound links (potentially mis-filed)
