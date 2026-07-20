---
type: paper
title: "Understanding World or Predicting Future? A Comprehensive Survey of World Models"
authors: ["Jingtao Ding", "Yunke Zhang", "Yu Shang", "Jie Feng", "Yuheng Zhang", "Zefang Zong", "Yuan Yuan", "Hongyuan Su", "Nian Li", "Jinghua Piao", "et al."]
year: 2024
venue: "arXiv 2024 (ACM Comput. Surv. preprint)"
url: https://arxiv.org/abs/2411.14499
rw_id: 01kx6cgmkkhp3cs8v569j0aav4
topics: [world-models, video-generation, reinforcement-learning, robotics, generative-models]
priority: medium
read_state: queued
relevance: ""
added: 2026-07-14
last_updated: 2026-07-16
---

## TL;DR

A wide-scope survey from Yong Li's group at Tsinghua that maps the world-model literature onto a single organizing question: should a world model **understand the present state of the world** (build internal representations of its mechanisms) or **predict its future** (simulate how it will evolve to guide decisions)? It traces this dichotomy back to two seminal definitions — Ha & Schmidhuber's recurrent world model (2018), which abstracts the world for decision-making, and LeCun's JEPA (2022), which insists a world model must also envision future states. From this frame it builds a two-branch taxonomy: (1) **implicit representation of the external world** (model-based RL, LLM/MLLM world knowledge), and (2) **future prediction of the physical world** (video-generation world models, embodied-environment simulators). It then surveys four application domains — generative games, embodied/robotics intelligence, urban intelligence (autonomous driving, logistics, analytics), and societal intelligence (social simulacra) — and closes with six open problems (physical rules, social dimension, benchmarks, sim-to-real, efficiency, ethics). Its value is as a **map**: it names the sub-categories and slots representative methods (Dreamer, TD-MPC2, Sora, Cosmos, Genie, GAIA-1, V-JEPA 2, WHAM, AgentSociety) into a coherent framework, and catalogs the benchmark landscape.

## Context & motivation

The survey is written against the 2024 surge of interest sparked by multimodal LLMs (GPT-4) and video generators (Sora), which appear to "capture aspects of world knowledge" — Sora's videos seem to obey physical laws — yet raise the question of whether they *truly* qualify as world models. The authors note the definition remains contested, "generally divided into two primary perspectives: understanding the world and predicting the future." They position their work against prior surveys, which they say fall into two camps: (1) application-specific reviews (video generation, autonomous driving, agents) and (2) reviews of the technological transition from multi-modal models to world models. Their stated gap: prior work "lacks a systematic examination of what precisely constitutes a world model and what different real-world applications require from these models." The three claimed contributions are: a novel two-function categorization system, a classification of how key application areas emphasize the two functions differently, and an outline of future directions. A companion paper/code list is maintained at `github.com/tsinghua-fib-lab/World-Model`.

## Method

### Problem formulation — how the survey defines a world model

The survey does not commit to one authored definition; it synthesizes a **shared consensus** across the literature: "the essential purpose of a world model is to understand the dynamics of the world and compute the next state with certainty (or with some guarantee), which empowers the model to extrapolate longer-horizon evolution and to support downstream decision-making and planning." It roots this in cognitive science — Craik's "the mind constructs small-scale models of reality" (1943) and Johnson-Laird's Mental Models Theory (1983), whose principles (finite representations of infinite domains; structural rather than superficial detail; predictive simulation of alternatives) it maps directly onto AI world models. The two anchoring technical definitions are Ha & Schmidhuber's RNN-based implicit model and LeCun's JEPA (encoder = perception module, predictor = cognitive module; self-supervised latent prediction rather than pixel prediction; dual System-1/System-2 framing).

### Core idea — the central dichotomy

The organizing axis is **understanding vs. predicting**, realized as two branches:

- **Implicit representation of the external world (Section 3):** transform external reality into a model of latent variables to enable informed decision-making — an *implicit comprehension* of world mechanisms. Includes both classic MBRL and the integration of world knowledge into LLMs.
- **Future prediction of the external world (Section 4):** generative models (primarily visual/video) that simulate future states, emphasizing the *realness* of generated futures; the survey tracks a transition "from visual to spatial representations and from video to embodiment."

The framing lets the survey then argue that each **application domain weights the two functions differently** (Section 5): autonomous driving needs both real-time perception (understand) and trajectory forecasting (predict); robotics needs implicit dynamics plus interactive embodied environments; social simulacra need abstract behavioral prediction.

### Architecture / algorithm — the full taxonomy

**Branch 1 — Implicit representation (Section 3)**

- **3.1 World model in decision-making**
  - **3.1.1 World model in model-based RL.** Formulates decision-making as an MDP $(\mathcal{S},\mathcal{A},M,r,\gamma)$; the world model is the transition dynamics $M$ (reward $r$ usually known), so the key task is learning $M$.
    - *World-model learning:* deterministic one-step MSE objective $\min_\theta \mathbb{E}_{s'\sim M^*(\cdot|s,a)}[\|s' - M_\theta(s,a)\|_2^2]$ (Eq. 1), where $M^*$ is the true dynamics and $M_\theta$ the learned model; and a probabilistic variant minimizing the KL divergence $\min_\theta \mathbb{E}_{s'\sim M^*}[\log(M^*(s'|s,a)/M_\theta(s'|s,a))]$ (Eq. 2, Chua et al. PETS — modeling aleatoric uncertainty). Both reduce world-model learning to supervised learning on trajectories ("simulation data"). Representation-learning line: Ha & Schmidhuber (autoencoder + latent state), Hafner's Dreamer series (DreamerV3 solves 150+ tasks incl. Minecraft diamond, no human data), Recall-to-Imagine (memory for long-horizon), next-token-prediction transformers (Janner, Schubert), PWM (Georgiev, large multi-task off-policy). Cites Richens et al. — any agent generalizing to multi-step goal-directed tasks must have learned a predictive model.
    - *Policy generation:* Model Predictive Control $\max_{a_{t:t+\tau}} \mathbb{E}[\sum_{t'=t}^{t+\tau} r(s_{t'},a_{t'})]$ (Eq. 3) with sampling schemes (Nagabandi Monte-Carlo, Chua trajectory-sampling ensemble, TD-MPC2 latent-space optimization); and Monte Carlo Tree Search (AlphaGo/AlphaGo Zero, value-prediction networks).
  - **3.1.2 World model with language backbone.** *Direct action generation via LLM world models* (Yang text-to-video for robot control, Zhou compositional world model, multi-expert VLN, omni-graph navigation, web agents WMA and WKM). *Modular usage* (Guan — GPT-4 generates/refines PDDL for off-the-shelf planners; Xiang E2WM in VirtualHome with MCTS; Dynalang multimodal world model; RAFA Bayesian-adaptive-MDP; WebDreamer; WorldCoder — world model as an editable Python program).
- **3.2 World knowledge learned by models** — three sub-buckets by spatial scope:
  - **3.2.1 Global physical world:** Gurnee & Tegmark (LLMs have linear "space neurons"/"time neurons"), GeoLLM/Manvi (prompt-extracted geospatial knowledge), AgentMove, GPS-to-Image, GLOBE; caveat that urban knowledge in LLMs is "coarse and inaccurate" (CityGPT).
  - **3.2.2 Local physical world:** cognitive-map concept (Tolman); Cornet & Thomson (visual predictive coding builds a spatial cognitive map from pixels in Minecraft), Dynalang, Jin (emergent program semantics from next-token prediction); benchmarks (WM-ABench, Spatial457, Thinking-in-Space) reveal a large gap even in simple local environments.
  - **3.2.3 Human society:** Theory of Mind (Premack & Woodruff); Sap, Strachan (GPT-4 strong but fails faux-pas detection); ethical/emotional judgment (Mozikov), cultural sensitivity (SafeWorld, 100-languages); enhancement methods COKE (ToM knowledge graph) and SimToM (two-stage prompting).

**Branch 2 — Future prediction (Section 4)**

- **4.1 World model as video generation**
  - **4.1.1 Towards video world models:** a video world model "simulates and predicts the future state of the world by processing past observations and potential actions within a visual context." Sora as candidate world simulator + its limits (weak causal reasoning; inconsistent physical laws). Base models OpenSora, CogVideoX, Wan; Cosmos (physics-adherence via massive real-physics video, diffusion + auto-regressive); Genie 2/3 (interactive/gaming, action-conditioned auto-regressive diffusion). Open technical challenges: long-duration generation, interactive generation, physical-law adherence.
  - **4.1.2 Capabilities of video world models** (what separates them from plain video generators): **long-term predictive ability**, **multi-modal integration** (image/action conditioning), **interactivity/controllability**, **diverse environments**. Table 2 organizes representative models by capability: Long-term (NUWA-XL, LWM, GAIA-1, StreamingT2V), Multimodal (3D-VLA, Pandora, Genie, UniSim), Interactive (VideoDecision, iVideoGPT, PEEKABOO, Aether, Yume, NWM), Consistency (WorldGPT, DiffDreamer, ConsistI2V, WorldMem), Diverse environments (WorldDreamer, MUVO, UniWorld).
- **4.2 World model as embodied environment** — three sub-categories (Table 3):
  - **4.2.1 Indoor:** AI2-THOR, Matterport3D, VirtualHome, Habitat, SAPIEN, iGibson (adds lidar), AVLEN (adds audio), GRUtopia (social/NPCs), LLM-driven flexible generation (Holodeck, AnyHome, LEGENT).
  - **4.2.2 Outdoor:** MetaUrban (asset-retrieval urban), UrbanWorld (3D-generative urban), MineDOJO (Minecraft sandbox).
  - **4.2.3 Dynamic (generative, first-person):** UniSim (robot-manipulation video from conditions), Pandora, AVID, EVA (adds VLM anticipation), Streetscapes (autoregressive video diffusion); physics-integrating trend: Aether (camera-trajectory RGB-D), TesserAct (normal maps), Roboscape (depth + keypoints), Deepverse (geometric predictions across timesteps).

**Application layer (Section 5)**

- **5.1 Game intelligence** — three capability dimensions: *interactivity* (GameNGen — neural DOOM at 20 fps; GameGen-X; Matrix-Game 17B), *consistency* (numerical/spatial; MineWorld; WHAM = World and Human Action Model), *generalization* (GameFactory, generative infinite games, uncertainty-driven exploration).
- **5.2 Embodied intelligence / robotics** (Table 4): *5.2.1 learning implicit representation* (CNN/ViT, RoboCraft GNN, PointNet, SpatialLM, LLM task representations BC-Z/Text2Motion); *5.2.2 predicting future states* — three uses: **synthetic data generation** (DreamGen neural trajectories for VLA, Roboscape, EVAC), **action guidance via imagined futures** (UniPi, VIPER, GR-2, VPP, Genie Envisioner, Vidar, V-JEPA 2 latent-space + MPC), **environment simulation for policy evaluation** (IRASim, GE-Sim, Roboscape); *5.2.3 sim-to-real* (NeBula belief space, DayDreamer real-world locomotion in hours, SWIM human-video + <30 min fine-tuning).
- **5.3 Urban intelligence** — *5.3.1 autonomous driving* (pipeline: perception/prediction = understanding; world simulators = prediction; Table 5: PointNet, BEVFormer, Transfusion, UniAD, TOKEN, OmniDrive for understanding; GAIA-1, DriveDreamer, Drive-WM, OccWorld/OccSora (occupancy), Vista, Copilot4D (radar point cloud) for simulation); *5.3.2 autonomous logistics* (micromobility Vid2Sim/CityWalker/URBAN-SIM/NWM; low-altitude aerial CityNavAgent/AirScape — first aerial world model); *5.3.3 urban analytics* (UrbanLLaVA, AgentMove, CAMS, PIGEON, GPS-to-Image).
- **5.4 Societal intelligence** — social simulacra as world models, two perspectives (Table 7): *mirroring real-world society* (Generative Agents/AI Town, S3, EconAgent, SRAP-Agent, Project Sid, OASIS, AgentSociety, SocioVerse) and *agents' understanding of external world* (Agent-Pro belief formation, GovSim collective cognition, AgentGroupChat).
- **5.5 Functions of world models** — a cross-cutting split: **cloud-based environments** (video generators as data engine / RL environment / policy evaluator) vs. **edge-side agent brains** (latent-space compression for on-device planning, e.g. V-JEPA 2 + MPC, or two-stage VPP).

### Derivations / why it works

_survey paper; no single derivation._ The only formalism introduced is the standard MBRL scaffolding reproduced above: the MDP tuple, the MSE and KL world-model-learning objectives (Eqs. 1–2), and the MPC planning objective (Eq. 3). These are restatements of known results, used to unify the decision-making sub-branch, not new theory.

### Training procedure

_not addressed by the source_ (survey; no single training recipe — it catalogs others' setups, e.g. DreamerV3 normalization tricks, Cosmos pretraining on real-physics video, but gives no unified procedure).

### Inference / sampling

_not addressed by the source_ beyond the taxonomy of decision-time planning it catalogs (MPC, MCTS for the RL branch; autoregressive vs. diffusion generation for the video branch).

## Experimental setup

Not an empirical paper — the "setup" is the corpus and the benchmark landscape it catalogs. Datasets/environments appear as taxonomy entries: embodied-environment platforms (Table 3: AI2-THOR, Matterport3D, Habitat, iGibson, MetaUrban, MineDOJO, EmbodiedCity, GRUtopia, UrbanWorld, and dynamic generators UniSim/Pandora/Streetscapes), autonomous-driving stacks (Table 5), robotics tasks (Table 4), and social-simulacra platforms (Table 7). Benchmarks are consolidated in **Table 8** across three families: **video-centric world simulation** (WorldSimBench, WorldScore, VBench, VBench-2.0, T2V-CompBench), **physical & causal/spatial reasoning** (PhysBench, UrbanVideo-Bench, Physics-IQ, T2VPhysBench, VideoPhy, Basic Spatial Abilities), and **embodied decision-making** (EAI/Embodied Agent Interface, EWMBench, WPE, RoboScape as data-engine evaluator).

## Key results

Main takeaways (the survey's synthesis, not new measurements):
- The field genuinely splits into "understanding" vs. "predicting" camps, and most confusion about "is X a world model?" dissolves once you ask which function X serves.
- **Data-driven scaling alone is insufficient to recover physical laws.** Diagnostic work it cites (Kang et al. — scaling yields perfect in-distribution fidelity but "case-based" not "rule-based" generalization; Motamed et al. Physics-IQ — visual realism decoupled from physical understanding) motivates **hybrid physics-embedding** approaches (Genesis, PhysGen, physics-informed diffusion with PDE residual losses).
- The frontier of embodied world models is **transitioning from static, pre-built environments to dynamic, generative, action-conditioned, first-person simulators**, increasingly with explicit physical constraints (depth, normals, camera trajectories).
- Video world models must add four capabilities beyond video generation: long-term prediction, multi-modal integration, interactivity, environment diversity.
- Benchmarking has no canonical task/metric because goals diverge — a recognized open problem.

## Ablations

_not applicable (survey)._

## Limitations

Survey's own stated open problems (Section 6): (1) **physical rules & counterfactual simulation** — models fail on OOD/combinatorial physics; hybrid explicit-simulator approaches needed; (2) **enriching the social dimension** — realistic human-behavior simulation and its scalable (non-subjective) evaluation remain open; (3) **benchmarks** — no standard protocol; need more diverse, generalization-testing testbeds; (4) **bridging sim-to-real for embodied intelligence** — closing sim-to-real gaps, self-reinforcing loops between generative world models and embodied agents; (5) **simulation efficiency** — autoregressive transformer generation is one-token-at-a-time (bad for high-fps drone control); big+small model mixtures, distillation, request scheduling, GNN physics approximation; (6) **ethics & safety** — data privacy (opaque training data, GDPR), simulating unsafe scenarios via adversarial prompting, accountability/deepfakes/watermarking.

Honest-reader flags [analyst's view]: the taxonomy's two branches are not clean partitions — most modern systems (V-JEPA 2, Cosmos, GAIA-1) do *both* understanding and prediction, so the "understanding vs. predicting" split is more a spectrum than a dichotomy, which the survey acknowledges only implicitly. Coverage skews toward the authors' own group's work (CityGPT, UrbanLLaVA, AgentMove, AgentSociety, S3, EmbodiedCity, CAMS all appear prominently). The urban/societal-intelligence sections are notably deeper than a neutral survey would be, reflecting the lab's focus. Reference numbering and some table formatting are noisy in this preprint.

## Why it matters [analyst's view]

This is the best available **single map** over the exact cluster of world-model work the vault has been collecting. It gives a vocabulary — "implicit representation" (understanding) vs. "future prediction," and within prediction the "video → embodiment / visual → spatial" transition — that cleanly locates the vault's individual paper notes. AdaWorld (action-conditioned latent world models) and RoboDreamer sit in the video-generation-for-robotics bucket (5.2.2, action guidance via imagined futures); PhyWorld belongs to the physical-law-adherence challenge (6.1); DINO-world-models and V-JEPA-style latent models are the "edge-side agent brain / implicit representation" line (3.2, 5.5); World4RL fits the MBRL branch (3.1.1); world-action models (WHAM-lineage) span game intelligence (5.1) and embodied prediction. Reading this note first, then the individual method notes, gives a two-level structure: framework here, mechanisms there. The benchmark catalog (Table 8) is directly useful for evaluating any future world-model paper.

## Open questions / things to verify

- The survey asserts data-driven scaling "is insufficient to recover robust physical laws" — worth checking against the primary sources (Kang et al., Motamed et al. Physics-IQ) rather than taking the survey's framing at face value; this is exactly the PhyWorld question.
- Is the two-branch taxonomy predictive or merely descriptive? Does knowing a system is "understanding-first" vs. "prediction-first" tell you anything about its failure modes? The survey implies yes but does not test it.
- The MBRL formalism (Eqs. 1–3) is standard; verify Richens et al.'s claim that "any agent that generalizes to multi-step goal-directed tasks must have learned a world model" — a strong theoretical statement cited in passing.
- Venue: this appears formatted for ACM Computing Surveys (footer "ACM Comput. Surv., Vol. 1, No. 1") but circulated as arXiv 2411.14499; confirm final publication status before citing.

## Connections

- Builds on / frames: [[papers/gao-2025-adaworld]], [[papers/zhou-2024-robodreamer]], [[papers/baldassarre-2025-dino-world-models]], [[papers/ye-2026-world-action-models]]
- Situates the physics-fidelity debate: [[papers/zhao-2026-phyworld]]
- MBRL branch: [[papers/jiang-2025-world4rl]]
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/reinforcement-learning]], [[topics/robotics]], [[topics/generative-models]]
- Author indices: [[authors/jingtao-ding]], [[authors/yong-li]]

## Selected quotes (optional)

> "the essential purpose of a world model is to understand the dynamics of the world and compute the next state with certainty (or with some guarantee), which empowers the model to extrapolate longer-horizon evolution and to support downstream decision-making and planning." — §2.3

> "the evidence suggests that data-driven scaling alone is insufficient to recover robust physical laws. Integrating explicit simulators – or otherwise enforcing physical priors – remains a promising path…" — §6.1

> "world models are regarded as tools for either understanding the present state of the world or predicting its future dynamics." — Abstract
