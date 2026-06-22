---
type: paper
title: "Robots Need More than VLA and World Models"
authors: ["Elis Karcini", "Faisal Mehrban", "Quang Nguyen", "Mac Schwager", "Arash Ajoudani", "César Cadena", "Jan Peters", "Marco Hutter", "Haitham Bou-Ammar"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2606.06556
rw_id: 01ktspa4bqxkrse60j83tcsw8a
topics: [robotics, world-models, vision-language-action, learning-from-video, reward-modeling]
priority: high
read_state: queued
relevance: ""
added: 2026-06-10
last_updated: 2026-06-10
---

# Robots Need More than VLA and World Models

## TL;DR

This is a position/survey paper from Motoniq.ai (with collaborators at Stanford, ETH Zurich, IIT, TU Darmstadt, UCL) arguing that generalist robotics is wrongly framed as a *policy-scaling* problem ("collect more demos, train bigger VLAs, expect generalisation"). The authors claim the central bottleneck is not policy learning but the **absence of mechanisms that convert the world's abundant unstructured behavioural data into grounded robot supervision**. Human motion, internet video, simulation rollouts, and demonstrations contain rich information about tasks, goals, contacts, failures, and physical constraints, but lack the embodiment-specific action labels, task semantics, and reward structure robot policies need. They reorganise the field around the *grounding bottleneck* each line of work exposes, and propose **four missing pillars**: (1) a **physical data engine** for embodied autolabelling, (2) **task-preserving retargeting** across embodiments, (3) **physics-grounded world models** for consequence prediction, and (4) **self-improving deployment loops** with task-conditioned reward grounding. VLAs are recast as *one layer* in a larger physical-intelligence stack, not the whole stack. The paper contains no experiments — it is a conceptual reframing plus a lightly formalised research agenda.

## Context & motivation

The framing slogan is: *"Robotics is entering its foundation-model moment, but it does not yet have its internet."* Text and images are abundant, digitised, and densely paired with human supervision; physical interaction is not. The world contains vast behavioural data (human demonstrations, internet videos, factory workflows, household activities, simulation rollouts) that reveal *what task is attempted, how objects move, when contact occurs, and whether an outcome succeeds*, but rarely provide *embodiment-specific action labels, force signals, task semantics, or reward structure*.

The authors stress a structural asymmetry between robot data and text corpora: "every trajectory must be physically executable, every action is tied to a particular body, and every failure may damage hardware, objects, or the environment." Hence usable robot supervision stays tiny relative to the physical behaviour already present in the world.

Their organising move is to switch the field's lens from a **robot-data-centric pipeline** (collect demos → attach labels → train policy → eval on hardware → repeat) to a **grounding-centric pipeline** (start from broad physical experience → pass through grounding mechanisms that produce robot-usable actions, contacts, object states, task phases, goals, rewards). The central question is reframed as: *not which architecture to train, but which mechanisms make the world's physical experience learnable by robots.* Explicitly, the paper says its argument is **not** that VLAs are unimportant — rather that they are a *policy interface* dependent on upstream grounding.

## Method

This is a position paper; its "method" is (a) a survey reorganised by supervision bottleneck, and (b) a lightly mathematically formalised four-pillar research agenda. I reconstruct both below.

### Problem formulation

The paper formalises the gap between **observed physical change** and **executable robot action**. A video gives an observation sequence $o_{1:T} = \langle o_1,\dots,o_T\rangle$ but not the robot action sequence $a_{1:T} = \langle a_1,\dots,a_T\rangle$. Robot-native imitation assumes access to pairs $(o_t, a_t)$; learning from human/internet video usually provides only $o_{1:T}$, sometimes with language $L_{1:T}$ or weak metadata. The latent variables sought are **action-like representations** $z_t$ that explain transitions:
$$z_t \sim q(\cdot \mid o_t, o_{t+1}, L_t, L_{t+1}).$$
These latent variables are **not yet tied to any robot embodiment** — they are "transition codes" or "physical-change descriptors" that become robot actions only when an embodiment-conditioned decoder maps them to executable commands.

### Core idea

The missing layer in robotics is **not another policy architecture**, but a set of components that transform physical experience into robot-usable supervision. The next foundation model for robotics "will not be only a VLA or a world-model. It will be a pipeline that grounds physical experience into actions, rewards, world models, and deployment feedback." The stack is explicitly **closed-loop, not feed-forward**: deployment is treated as a supervision source, not just evaluation.

### Framework / taxonomy

**Survey reorganisation (Section 2) — by supervision bottleneck.** The authors decline to organise by data source or algorithmic family; instead each cluster is keyed to the grounding bottleneck it exposes:

- **2.1 Robot-native supervision** — data already in the robot's coordinate system (observations paired with embodiment-specific actions, task labels, rewards). Surveyed: datasets (RoboNet, BridgeData V2, DROID ~76k trajectories/350h, RH20T 110k+ contact-rich sequences, Open X-Embodiment/RT-X 1M+ trajectories across 22 embodiments); generalist policies (BC-Z, RT-1 ~130k episodes/700+ tasks, RT-2, SayCan, PaLM-E, Octo 800k trajectories, RoboCat, Dobb-E); policy machinery (Diffusion Policy, ALOHA/ACT); VLAs (Gato, VIMA, RoboFlamingo, OpenVLA 7B/~970k demos, π0 flow-matching, CogACT, RoboMamba, FAST tokenisation); spatial VLAs (SpatialVLA ~1.1M episodes, RDT-1B, 3D-VLA, 3DS-VLA, GeoVLA, GraphCoT-VLA, Avi); World-Action Models (DreamZero, UVA); humanoid VLAs (NVIDIA Isaac GR00T N1 dual-system, Gemini Robotics, Figure Helix, LeVERB, WholeBodyVLA, HuMI, HEX, Skild Brain). **Takeaway: strength = limitation** — these work *because* the data is already grounded, which is exactly the scaling ceiling.

- **2.2 Learning from weakly grounded physical observations** — passive video has abundant behavioural evidence but weak grounding. A four-way taxonomy of weak supervision video can provide: (i) **visual representations** (R3M, VIP, MVP, VC-1); (ii) **latent action codes** (LAPA — VQ-VAE over video transitions then a latent VLA; UniVLA — task-centric latent actions across embodiments/viewpoints); (iii) **task-progress / reward signals** (PROGRESSOR, Adapt2Reward, ReWiND, TimeRewarder, SARM); (iv) **behavioural priors** (affordances, contact, temporal structure). Earlier cross-embodiment work (TCN, AVID, XIRL, DVD) is cited as exposing the **correspondence problem** in explicit form. Takeaway: a latent action is not a command; a progress signal is not a reward; a human strategy may not be executable — video relocates the grounding problem, it does not remove it.

- **2.3 Generating physical experience** — four sub-routes for manufacturing experience: (a) **Simulation** (RLBench, Meta-World, ManiSkill, CALVIN, LIBERO) — but the designer pre-specifies state/action/success; (b) **Data generation** (MimicGen 50k+ demos from <200 seeds, RoboCasa, RoboCasa365, RoboGen) — open question is whether variations preserve contact/friction/failure modes; (c) **Real-to-sim-to-real** (RialTo digital twins, 3DGS-based RL-GSBridge / Real-is-Sim / RoboGSim, navigation transfer via SOUS VIDE / SINGER / GRaD-Nav, plus domain randomisation, ANYmal/RMA legged transfer); (d) **World modelling** (Schmidhuber's differentiable world, Ha & Schmidhuber World Models, PlaNet/Dreamer/DreamerV3/DayDreamer; robot-specific RoboDreamer, UniSim, Genie; object-centric/3D FOCUS, PointWorld, ParticleFormer; uncertainty-aware world models; physics-structured Deep Lagrangian/Hamiltonian/Lagrangian Nets, Interaction Networks, graph-network simulators; JEPA family I-JEPA/V-JEPA/V-JEPA 2; neural-scene+physics hybrids Physically Embodied Gaussian Splatting, Gaussian World Models, ContactGaussian-WM, PIN-WM). Takeaway: generated experience is useful **only when it preserves the physical variables that determine control** — visual realism alone is insufficient; world models *sharpen* rather than solve the grounding problem.

### Derivations / argument structure (the four pillars, Section 3)

The load-bearing content is the formalised four-component pipeline. Each pillar feeds the next, and deployment closes the loop.

**Pillar 1 — Physical data engine & embodied autolabelling (§3.1).** Treat physical experience as *partially labelled interaction data*, not raw video. A raw heterogeneous, asynchronous episode is:
$$\mathbf{x} = \{(v_i,\tau_i^{(v)})_{i=1}^{T_v},\,(m_j,\tau_j^{(m)})_{j=1}^{T_m},\,(h_k,\tau_k^{(h)})_{k=1}^{T_h},\,(r_l,\tau_l^{(r)})_{l=1}^{T_r},\,\mathbf{L}\}$$
where $v$ = timestamped video frames, $m$ = motion-capture / wearable / body-pose readings, $h$ = tactile/force/contact/hand-sensor readings, $r$ = raw robot logs (proprioception, deployment metadata), $\mathbf{L}$ = associated language (instruction/caption/correction). Not every episode has all modalities; in general $\mathbf{x} \in \mathcal{X}$, the space of heterogeneous physical episodes.

Because streams are asynchronous, the first hidden object is a temporal **alignment** to a common latent event timeline $\zeta \in \{1,\dots,Z\}$ via an alignment map
$$A: \{\tau_i^{(v)}, \tau_j^{(m)}, \tau_k^{(h)}, \tau_l^{(r)}\} \to \{1,\dots,Z\},$$
e.g. "video frames 30–55, motion readings 102–180, tactile spike @1.8s" all map to $\zeta=2$: `contact-begins`. Alignment is *part of* the autolabelling problem, not preprocessing.

For each event the engine recovers latent structure
$$\mathbf{z}_\zeta = [\mathbf{s}_\zeta, \mathbf{c}_\zeta, \boldsymbol{\phi}_\zeta, \mathbf{u}_\zeta, \mathbf{r}_\zeta],$$
where $\mathbf{s}_\zeta$ = object-centric physical state, $\mathbf{c}_\zeta$ = contact/interaction label, $\boldsymbol{\phi}_\zeta$ = task phase, $\mathbf{u}_\zeta$ = latent physical action / transition code, $\mathbf{r}_\zeta$ = task-conditioned progress/reward. At episode level it also infers a goal $g$ and outcome $y$ (success / failure / partial / unsafe), so the full hidden explanation is $\mathbf{z} = [\mathbf{z}_{1:Z}, g, y]$. The engine is an **inference model** $q_\theta(\mathbf{z}, A \mid \mathbf{x})$ that *jointly* solves alignment, event segmentation, object-state estimation, contact inference, phase recognition, latent-action discovery, reward grounding, and outcome prediction. Key conceptual distinction: a captioning model says "a person places a cup on a tray"; an *embodied autolabeller* recovers the physical event sequence (`reach-to-cup` → `contact-begins` → `grasp` → …) that can be retargeted, simulated, rewarded, or trained on. The authors stress these labels are *coupled* (contact → object-state → task progress → reward → action), and that wearable/mocap suits are a "labelling instrument for the physical world", with a dual purpose: teaching robots about *tasks* and about *humans* (for human-aware, human-compliant, collaborative policies). Central open problem: learn $\mathbf{x} \to (\mathbf{z}_{1:Z}, g, y, A)$ when most episodes give only partial supervision.

**Pillar 2 — Task-preserving retargeting across embodiments (§3.2).** Inferring events does not produce a policy — the **embodiment gap** remains (human hand vs parallel-jaw gripper vs dexterous hand vs mobile manipulator vs quadruped vs humanoid differ in kinematics, dynamics, sensors, action spaces, contact surfaces, failure modes). The goal is not to copy human motion but to **preserve the task-relevant physical effect**. For embodiment $e$, retargeting seeks an executable action/skill
$$a_\zeta^{(\text{embodied})} = f_\psi(\mathbf{u}_\zeta, \mathbf{s}_\zeta, \text{embodiment})$$
such that the robot-induced transition preserves the goal-relevant change:
$$\Delta_g(\mathbf{s}_\zeta, a_\zeta^{(\text{embodied})}) \approx \Delta_g(\mathbf{s}_\zeta, \mathbf{u}_\zeta),$$
where $\Delta_g$ is the task-relevant effect under goal $g$ (drawer displacement for opening, object pose for placing, alignment for insertion, containment for packing, contact state for grasping). This yields an **invariance hierarchy**, which generalist robotics must climb: pose-preserving (map human hand/arm to end-effector) → contact-preserving (touch right surfaces at right moments) → object-state-preserving (drawer opens, cup lifts, peg aligns) → intent/skill-preserving (different motion entirely, same task under same constraints). This also re-justifies wearable sensing: a suit need not give the final robot action, only the intermediate transferable variables (contacts, force events, object-state changes, phase boundaries, latent actions).

**Pillar 3 — Physics-grounded world models for consequence prediction (§3.3).** A candidate action is useful only if the robot can anticipate its consequence. The world model performs **consequence prediction**: given object-centric state $\mathbf{s}_\zeta$, goal $g$, and a latent action $\mathbf{u}_\zeta$:
$$\mathbf{s}_{\zeta+1} \sim p_\omega(\cdot \mid \mathbf{s}_\zeta, \mathbf{u}_\zeta, g),$$
or for a specific embodiment with action $a_\zeta^{(\text{embodied})}$:
$$\mathbf{s}_{\zeta+1} \sim p_\omega(\cdot \mid \mathbf{s}_\zeta, a_\zeta^{(\text{embodied})}, \text{embodiment}, g).$$
The first form supports task-level reasoning ("what should happen if the action is *pull / lift / insert / place*?"); the second supports embodiment-specific planning. The central design claim: **consequence prediction should be task-conditioned** — the model need not predict every pixel equally well, only the variables relevant to the task (drawer displacement + handle contact for opening; liquid state + container pose for pouring; deformable geometry + contact points for folding). Evaluation question shifts from "does the future look realistic?" to "does the prediction preserve the physical consequences that determine success or failure?" Physics grounding (object permanence, geometric consistency, conservation laws, differentiable contact, material parameters, action-conditioned dynamics, uncertainty estimates) reduces failure modes where purely learned predictors exploit visual regularities (interpenetration, contact without force, unrealistic deformation). The world model has four uses: *before* execution (evaluate retargeted candidates), *during* planning (search alternatives), *after* failure (explain what went wrong), *during* training (generate counterfactuals). Representation choice is left open (pixel / object-centric / 3D point-cloud-mesh-field-Gaussian / mechanics-based), with the authors favouring **hybrids** (3D scene + object-centric structure + physics constraints + data-driven residual dynamics).

**Pillar 4 — Self-improving deployment loops & reward grounding (§3.4).** After execution, the question is whether the outcome was *useful*, requiring **task-conditioned reward grounding**. Reward is written
$$r_\eta(\mathbf{s}_\zeta, g, \boldsymbol{\phi}_\zeta),$$
with $\mathbf{s}_\zeta$ the inferred state, $g$ the goal, $\boldsymbol{\phi}_\zeta$ the task phase. Reward is "an interpretation of physical progress under a goal", not an intrinsic scalar (a cup on a table = success for "put down", failure for "pick up", irrelevant for "open drawer"). The deployment loop is:
> deploy policy → observe outcome → infer task-conditioned progress / success / failure → explain failure or correction → add grounded supervision to the data engine → update reward model, world model, retargeting, and policy → redeploy.
A key requirement is **component-level credit assignment**: route supervision to the *right* component — policy updates for poor actions, world-model updates for wrong consequence predictions, retargeting updates when the physical effect was not preserved, reward-model updates when success/failure was misclassified. Without this, "the system may know that a rollout failed but not what should change." This closes the pipeline: data engine → retargeting → world model → reward grounding → deployment supplies new episodes back into the data engine, producing a *compounding physical-intelligence system*.

### Details specific to this paper

- The author affiliations indicate this is a **Motoniq.ai** (industry) position paper — five of nine authors list Motoniq.ai, with academic co-authors (Schwager/Stanford, Ajoudani/IIT, Cadena & Hutter/ETH Zurich, Peters/TU Darmstadt, Bou-Ammar/UCL Centre for AI). [analyst's view: reads as a research-agenda/recruiting manifesto for an applied robot-foundation-model company, which colours the framing.]
- It explicitly cites Schmidhuber's 1990 "making the world differentiable" work as the historical root of learned world models.
- It singles out **V-JEPA 2** as "one of the clearest recent links between JEPA-style world models and embodied control" (combines internet-scale video + small robot interaction data, reports zero-shot robot control).
- It raises **world-model uncertainty calibration** as an under-developed but important sub-area (Mei et al. 2025 VAE latent UQ; Li et al. 2025a UQ for RL-environment world models; Ward et al. 2026 calibrated-latent world model used to detect runtime errors in a VLA manipulation policy).

## Experimental setup

_Not addressed by the source._ This is a position/survey paper with no experiments, datasets, baselines, or metrics of its own. All numbers in the note (DROID 76k, RT-1 130k, OpenVLA 970k, MimicGen 50k, etc.) are figures the paper *quotes from the surveyed works*, not results the authors produced.

## Key results

No empirical results. The paper's "result" is the conceptual contribution: (1) reframing generalist robotics from policy scaling to a **grounding** problem; (2) a survey reorganised by *supervision bottleneck* (robot-native → weak video → generated experience); (3) the **four-pillar agenda** (physical data engine / task-preserving retargeting / physics-grounded world models / self-improving deployment loops) with light mathematical formalism; (4) a proposed **closed-loop compounding pipeline**; (5) a reframed evaluation agenda — ask whether a system can convert *weaker* sources of experience into supervision, not just whether a bigger policy solves more tasks.

## Ablations

_Not addressed by the source._

## Limitations

Paper's own acknowledged limits:
- World-model uncertainty quantification "is still a young area"; learned world models are subject to OOD errors/hallucinations that can create a "vicious cycle" in planning — they flag this as unsolved rather than resolved.
- The latent variables learned from video are "better understood as transition codes or physical-change descriptors until they are grounded in a specific robot embodiment" — i.e. the paper acknowledges its own central object (the latent action $\mathbf{u}_\zeta$) is not yet operational.
- The representation choice for physics-grounded world models is left explicitly open.

[analyst's view] flags:
- The four pillars are **specified, not built** — every formal object ($q_\theta$, $f_\psi$, $p_\omega$, $r_\eta$, the alignment $A$) is a desideratum with no instantiation, training recipe, or feasibility result. The joint inference $q_\theta(\mathbf{z}, A \mid \mathbf{x})$ over alignment + segmentation + state + contact + phase + latent action + reward + outcome is enormously underconstrained and the paper offers no learning signal for most of those latents on action-free data.
- The "task-preserving retargeting" invariance hierarchy is intuitive but the strongest (intent-preserving) level is essentially the full manipulation problem restated; calling it "retargeting" may understate its difficulty.
- As an industry position paper, the survey is broad but necessarily selective; it leans on conceptual "takeaways" rather than head-to-head evidence about which grounding mechanism actually works best.

## Why it matters [analyst's view]

The paper is a useful *map* of robot-learning circa 2026 and a clean articulation of an intuition that the vault's other notes also gesture at: that **world models are only valuable to robotics insofar as they preserve control-relevant physical structure**, not visual plausibility. That thesis directly echoes [[papers/joseph-2026-physics-video-world-models]] (physics is encoded distributedly in video world models — plausibility ≠ grounded physics), [[papers/higuera-2026-visuo-tactile-world-models]] (adding tactile grounding fixes physics-violation failures of vision-only models), and [[papers/maes-2026-leworldmodel]] (JEPA-style latent prediction that encodes recoverable physical structure for planning). The paper's elevation of V-JEPA 2 and the JEPA family situates it squarely next to the vault's [[topics/jepa]] and [[topics/self-predictive-learning]] threads. For someone tracking the world-models lineage, this paper is the "robotics-stack" framing that contextualises why the vault's world-model notes matter for embodiment. It is also a good source of a structured reading list — most of the surveyed systems are candidates for their own deep notes.

## Open questions / things to verify

- Is the arXiv id `2606.06556` correct? The text's source URL is `arxiv.org/pdf/2606.06556`; the date (year 2026) is consistent but the id should be re-checked when citing.
- Does any of the cited "future" work (Ward et al. 2026 calibrated-world-model runtime error detection for VLAs; HuMI/HEX humanoid systems) already partially instantiate the four pillars? Worth chasing to see how far the agenda is from reality.
- The joint inference model $q_\theta(\mathbf{z}, A \mid \mathbf{x})$ — is there any existing system that *jointly* recovers alignment + the full $\mathbf{z}_\zeta$ tuple, or is every component currently a separate model? The paper implies the latter.
- LAPA and UniVLA are the closest existing realisations of the "latent action" idea — worth a deep note to evaluate whether their latent codes actually transfer across embodiments as the pillar requires.

## Connections

- Builds on (conceptually): Ha & Schmidhuber World Models, the JEPA line (I-JEPA / V-JEPA / V-JEPA 2), LeCun's "Path Towards Autonomous Machine Intelligence" — all surveyed as antecedents.
- Related: [[papers/joseph-2026-physics-video-world-models]] (video world models encode physics distributedly), [[papers/higuera-2026-visuo-tactile-world-models]] (tactile grounding fixes vision-only physics failures), [[papers/maes-2026-leworldmodel]] (stable JEPA world model with recoverable physical latent), [[papers/ding-2024-diffusion-world-model]] (one-pass trajectory world model for offline RL), [[papers/tong-2026-beyond-language-modeling]] (world-modeling from unified multimodal pretraining).
- Contrasts with: pure policy-scaling / "bigger VLA" framing it explicitly argues against.
- Topic MOCs: [[topics/robotics]], [[topics/world-models]], [[topics/jepa]], [[topics/self-predictive-learning]]
- Author indices: [[authors/elis-karcini]], [[authors/faisal-mehrban]], [[authors/quang-nguyen]], [[authors/mac-schwager]], [[authors/arash-ajoudani]], [[authors/cesar-cadena]], [[authors/jan-peters]], [[authors/marco-hutter]], [[authors/haitham-bou-ammar]]

## Selected quotes

> "The central bottleneck is not only policy learning, but the absence of mechanisms that convert the world's abundant unstructured behavioural data into grounded robot supervision." — Abstract

> "Robotics is entering its foundation-model moment, but it does not yet have its internet." — §1 Introduction

> "We believe that the next foundation model for robotics will not be only a VLA or a world-model. It will be a pipeline that grounds physical experience into actions, rewards, world models, and deployment feedback." — §3

> "The correct retargeting target is not the human joint trajectory, but the physical transformation that matters for the task." — §3.2

> "The question is not 'does the future look realistic?' but 'does the prediction preserve the physical consequences that determine success or failure?'" — §3.3

> "Robots, therefore, need more than VLAs. They need the architectural pillars that make physical experience usable." — §4 Conclusions
