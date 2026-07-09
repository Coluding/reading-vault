---
type: paper
title: "DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning"
authors: ["Zhenyu Jiang", "Yuqi Xie", "Kevin Lin", "Zhenjia Xu", "Weikang Wan", "Ajay Mandlekar", "Linxi Fan", "Yuke Zhu"]
year: 2024
venue: "arXiv preprint"
url: https://arxiv.org/abs/2410.24185
rw_id: 01kwy6sdcmhs614mq4rt8v89cr
topics: [robotics, imitation-learning, manipulation, data-generation]
priority: high
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

DexMimicGen (DexMG) is an automated data-generation system that turns a *handful* of human teleoperation demos into a large imitation-learning dataset for **bimanual dexterous** robots (two arms + multi-fingered hands, e.g. humanoids). It extends MimicGen — which transforms and replays object-centric subtask segments in simulation for a single arm — to the two-arm setting by introducing a **taxonomy of subtask types** (parallel, coordination, sequential) and per-arm asynchronous execution, synchronization, and ordering-constraint mechanisms to handle multi-arm interaction. From just **60 source human demos** it generates **21K demos** across nine simulation tasks over three embodiments, and shows large policy gains (e.g. Drawer Cleanup 0.7% → 76.0%, Threading 1.3% → 69.3%, Piece Assembly 3.3% → 80.7%). A real2sim2real pipeline with a digital twin trains a real humanoid can-sorting policy to **90% success from 40 generated demos vs 0% from 4 source demos**.

## Context & motivation

Imitation learning from human demos is effective, but **data acquisition is the bottleneck** — teleoperation for a single arm already needs teams of operators and months of effort. Bimanual dexterous robots (humanoids) make this dramatically worse: simultaneously controlling two arms and two multi-fingered hands imposes high operator burden, requires special teleop interfaces, and needs *more* data because of the increased DoF and task complexity.

Automated data generation in simulation is the scalable alternative, proven for single-arm manipulation by **MimicGen** [Mandlekar et al., CoRL 2023]. MimicGen assumes a task decomposes into object-centric subtasks, each manipulation being relative to one object's frame; it segments source demos into these subtasks, applies a rigid SE(3) transform to adapt each segment to a new object pose, interpolates to connect segments, replays them open-loop, and keeps only successful rollouts. But MimicGen's *single* subtask segmentation can't express bimanual structure: the two arms may act independently, must sometimes coordinate tightly, and sometimes must obey a strict order. DexMimicGen's contribution is exactly the machinery to handle those three cases.

## Method

### Problem formulation
Each task is a POMDP; given $N$ demonstrations $D = \{(s^i_0, o^i_0, a^i_0, \dots, s^i_{H_i})\}_{i=1}^{N}$ with states $s$, observations $o$, actions $a$, learn a policy $\pi: O \to A$ via **Behavioral Cloning** with the maximum-likelihood objective $\arg\max_\theta \mathbb{E}_{(s,o,a)\sim D}[\log \pi_\theta(a \mid o)]$. Policies are trained on DexMG-generated data, not directly on source demos.

**Assumptions (inherited from MimicGen).** (A1) the per-arm action space = an end-effector pose command + a hand actuation command (1-D open/close for a parallel-jaw gripper, 6-D joint command for a dexterous hand); (A2) each task divides into object-centric subtasks; (A3) an object's pose is observable/estimable before an arm contacts it.

### Core idea
Generalize MimicGen's transform-and-replay from one arm to two by giving **each arm its own sequence of object-centric subtasks** and adding coordination machinery. The engine exploits **SE(3) equivariance of robot actions w.r.t. object poses**: if an object's pose gets an SE(3) transform, applying the same transform to the robot actions reproduces the original effect on the new pose. A subtask $S_i(o_i)$ manipulates relative to object $o_i$'s frame; a source segment $\tau_i = (T^{C_0}_W, T^{C_1}_W, \dots, T^{C_K}_W)$ is a sequence of end-effector poses in world frame $W$, transformed by the constant $T^{o'_i}_W (T^{o_i}_W)^{-1}$ so end-effector-to-object relative poses are preserved in the new scene.

### Architecture / algorithm — the three subtask types
1. **Parallel subtasks** (arms act independently, may finish at different times — e.g. each arm grasps a separate piece in Piece Assembly). Each task is a *per-arm* subtask sequence $S^{a_1}_1(o_1),\dots,S^{a_1}_{M_1}$ and $S^{a_2}_1(o_1),\dots,S^{a_2}_{M_2}$, and each arm's source demo is split into its own object-centric segments $\{\tau^n_i\}$, $n\in\{1,2\}$. Because arm subtasks aren't time-aligned, DexMG uses an **asynchronous execution strategy**: maintain an action queue per arm, dequeue both in parallel, and whenever an arm's queue empties, refill it with the next transformed subtask segment. No cross-arm alignment required.

2. **Coordination subtasks** (arms must maintain a precise *relative* pose — e.g. placing a lid with both hands, lifting a tray). Two requirements: (1) both arms execute in a **synchronized** manner, and (2) both trajectories use the **same** SE(3) transform. Temporal alignment is enforced by making coordination subtasks *end at the same timestep* during segmentation; at execution each arm waits until both have the same number of remaining steps. Two transform schemes are offered: **Transform** (use $T^{o'_i}_W (T^{o_i}_W)^{-1}$ computed at the moment the first arm begins the subtask) and **Replay** (use the source trajectory directly with no transform — beneficial for handovers because it keeps the trajectory within kinematic limits and fully executable).

3. **Sequential subtasks** (strict order — e.g. in Pouring, pour the ball into the bowl *before* moving the bowl to the pad). An **ordering-constraint mechanism** specifies a pre-subtask and a post-subtask; the arm doing the post-subtask waits until the other arm's pre-subtask completes. Using *different* source demos per arm needs this constraint but increases data diversity; using the *same* source demo often satisfies ordering automatically.

**Overall workflow** (Tray Lift example): segment source demos into per-arm subtasks via heuristics or human annotation and record reference-object poses; at generation time randomize the scene, pick a source demo, compute the object-relative transform, transform both arms' trajectories, execute with the appropriate (async / synchronized / ordered) strategy, and keep the demo only if a per-task success check passes. Finger motion is generated by **replaying source finger-joint actions** (finger movement is always relative to end-effector movement).

### Derivations / why it works
_no derivation; empirical/systems paper._ The method rests on the SE(3)-equivariance argument (same transform on object and actions preserves the interaction) rather than a formal proof.

### Training procedure
Environments built in **RoboSuite** with **MuJoCo** physics. Three embodiments: (1) bimanual Panda arms + parallel-jaw grippers, (2) bimanual Panda arms + dexterous hands, (3) a **Fourier GR-1 humanoid** + dexterous hands. Controllers: **Operational Space Control (OSC)** for Panda arms (delta EE pose → joint torques); an **Inverse Kinematics** controller based on `mink` for the humanoid (global EE pose → joint positions, handling the shared-torso kinematic tree); direct joint-position control for fingers. Teleoperation: iPhone-based (RoboTurk) for parallel-jaw; **Apple Vision Pro** (VisionProTeleop) for dexterous hands, with a fixed-pose human-to-robot calibration and **OmniH2O** retargeting for fingers. 10 source demos/task collected for grippers, only 5/task for dexterous hands (higher operator burden); DexMG then generates **1000 demos/task**. Policies: BC-RNN, BC-RNN-GMM, and **Diffusion Policy**.

### Inference / sampling
Standard visuomotor policy rollout (image-based observations → actions); DexMG itself is offline data generation, not a runtime component. Evaluation follows prior convention: 3 seeds, take the max success rate per seed.

## Experimental setup

- **Datasets:** 21K generated demos over 9 tasks × 3 embodiments; per-task datasets of 1000 demos (also 100/500/1000/5000 for the size study). Task variants D0/D1/D2 broaden the initial reset distribution.
- **Tasks:** Piece Assembly, Threading, Transport, Box Cleanup, Drawer Cleanup, Tray Lift, Pouring, Coffee, Can Sorting — spanning high-precision, articulated-object, and long-horizon manipulation with parallel/coordination/sequential structure.
- **Policies/baselines:** Diffusion Policy (DP), BC-RNN-GMM, BC-RNN; a **Demo-Noise** data-generation baseline (replay source demos with action noise); source-demos-only baseline.
- **Metric:** task success rate (%), 3 seeds.

## Key results

- **Huge gains over source-only** across all tasks (Table I, 1000-demo datasets). Standout jumps: Drawer Cleanup **0.7 → 76.0**, Threading **1.3 → 69.3**, Piece Assembly **3.3 → 80.7**, Pouring **0.7 → 79.3**, Can Sorting **0.7 → 97.3**. Best per-task success is generally 70–97%.
- **Beats Demo-Noise by >58%** on every compared task (e.g. Tray Lift 16.7 → 75.3; Pouring 26.7 → 79.3), and unlike Demo-Noise it can generate for broader initial distributions D1/D2.
- **Broader-distribution generalization:** policies trained on D1/D2 data stay performant under matching broad evaluation, though success degrades somewhat as the distribution widens (e.g. Piece Assembly 74.0/67.3/44.0 across D0/D1/D2).
- **Cross-benchmark:** applied to **BiGym** (humanoid mobile bimanual benchmark), generating 1000 demos each for three tasks with generation success rates 29.1% / 43.6% / 76.4%.
- **Real world:** on the GR-1 humanoid Can Sorting task, a Diffusion Policy trained on **40 DexMG demos → 90% success**, vs **0% from 4 source demos** (10 trials each for red/blue cups).

## Ablations

- **Dataset size:** clear gains from 100 → 500 → 1000 demos, but **diminishing (sometimes negative) returns from 1000 → 5000**, task-dependent.
- **Replay vs Transform (coordination/handover):** Replay wins on Transport (63.3 vs 46.0) and ties on Can Sorting (97.3 vs 98.6) — Replay is the default for handovers.
- **Ordering constraints (sequential):** consistently help — Drawer Cleanup 50.7 vs 48.0, Pouring 88.7 vs 76.7; using the same source demo for both arms gives 56.7 / 79.3 respectively.
- **Policy architecture:** Diffusion Policy generally best; notably **BC-RNN-GMM underperforms BC-RNN and DP**, especially with dexterous hands — *contradicting* the RoboMimic finding that a GMM head helps.
- **Distribution coverage (PCA):** DexMG substantially *expands* end-effector action coverage but for finger-joint actions mostly does *local interpolation* rather than broad expansion.

## Limitations

- **No explicit collision handling** — some generation failures come from generated-trajectory/object collisions; authors plan to add motion planning from SkillMimicGen.
- **Relies on MimicGen's assumptions** (object-centric subtasks; observable object poses pre-contact) and on manual specification of which subtask pairs are coordination/sequential.
- **Occlusion hurts vision policies** — e.g. Threading stays below 70% because the object and hole are occluded from the third-person camera; authors suggest active perception / visual RL.
- Generation success rates on hard benchmarks can be low (BiGym FlipCup 29.1%), meaning many attempts are wasted.
- **Open-loop replay** of transformed trajectories means the underlying skills are not reactive within a subtask.

## Why it matters [analyst's view]

This is the natural and well-executed extension of the MimicGen "transform-and-replay in sim" recipe into the two-arm/dexterous regime that everyone chasing humanoids needs. The conceptual contribution is small but genuinely useful: recognizing that bimanual tasks factor into parallel / coordination / sequential subtasks, and giving each a concrete execution mechanism (async queues, synchronized waits, ordering constraints). The strongest practical result is the **real2sim2real digital-twin pipeline** — 4 human demos → 40 sim-generated real-executable demos → 90% real policy — which is a credible template for bootstrapping humanoid manipulation without armies of teleoperators. The most interesting *scientific* nugget is the finding that GMM action heads hurt in the dexterous setting (contradicting RoboMimic), plus the PCA observation that finger actions get interpolated rather than expanded — hinting that hand dexterity, not arm placement, is the real coverage bottleneck. Because the datasets/envs are released, this is likely to become a standard substrate for bimanual IL studies.

## Open questions / things to verify

- The finger-action coverage stays narrow — does this cap the dexterity of learned policies, and would a hand-space augmentation scheme help?
- Why does GMM specifically fail with dexterous hands? Multimodality of finger actions vs arm actions?
- How much does open-loop replay limit robustness vs a closed-loop generator (e.g. with motion planning / reactive control)?
- Real-world results are one task (Can Sorting) — how well does the 90% transfer generalize to more contact-rich or long-horizon real humanoid tasks?
- Generation success rate (yield) vs task complexity — what determines when the transform-and-replay recipe breaks down?

## Connections

- Topic MOCs: [[topics/robotics]], [[topics/imitation-learning]], [[topics/manipulation]], [[topics/data-generation]]
- Author indices: [[authors/zhenyu-jiang]], [[authors/ajay-mandlekar]], [[authors/linxi-fan]], [[authors/yuke-zhu]]
- Builds on: MimicGen (Mandlekar et al., CoRL 2023); related to SkillMimicGen; policies from RoboMimic (Mandlekar et al., CoRL 2021) and Diffusion Policy (Chi et al., RSS 2023). No existing vault paper notes obviously linked yet.

## Selected quotes

> "The core idea is to leverage a small set of human demonstrations and use demonstration transformation and replay in physical simulation to automatically generate large amounts of training data suitable for imitation learning in the bimanual dexterous manipulation setting." — §I Introduction

> "The policy trained on DexMimicGen data achieves 90% success, while the model trained on the source data achieves 0%." — §VI-D Real-World Evaluation

> "DexMimicGen significantly expands the distribution coverage of end-effector actions. In contrast, for finger joint actions, DexMimicGen primarily performs local interpolation rather than broad expansion." — §X Result Analysis
