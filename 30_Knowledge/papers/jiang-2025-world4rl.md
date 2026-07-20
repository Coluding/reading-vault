---
type: paper
title: "World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation"
authors: ["Zhennan Jiang", "Kai Liu", "Yuxin Qin", "Shuai Tian", "Yupeng Zheng", "Mingcai Zhou", "Chao Yu", "Haoran Li", "Dongbin Zhao"]
year: 2025
venue: "arXiv preprint"
url: https://arxiv.org/abs/2509.19080
rw_id: 01kwy6myrsrz4djw5rvq1ggefc
topics: [world-models, diffusion-models, reinforcement-learning, robotics]
priority: medium
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---


# World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation

## TL;DR

**World4RL** is a two-stage framework that uses a **diffusion-based world model as a high-fidelity, frozen simulator** in which to refine an imitation-learning policy with reinforcement learning — entirely in imagination, with **no online real-world or simulator interaction** during RL. Stage 1 pretrains three components: a diffusion transition model (predicts next observation from a history of observations+actions), a binary reward classifier (sparse success signal from a ResNet18), and a BC policy. Stage 2 freezes the world model and refines the policy with **PPO** on imagined rollouts. Two design choices are central: a **two-hot action encoding** (lossless, differentiable continuous-action representation borrowed from DreamerV3) that bridges the RL agent and the world model, and a **diffusion (EDM/U-Net) backbone** that gives sharper, temporally coherent rollouts than RSSM/VAE world models (e.g. DiWA). It reports **+16% absolute** success-rate gain in simulation (Meta-World, 51.5%→67.5% over BC) and **+25%** on real Franka robots (68.3%→93.3%), and matches offline-to-online RL baselines (RLPD, Uni-O4) that need 300k+ online steps, while using zero online samples.

## Context & motivation

Robotic manipulation policies are usually bootstrapped by **imitation learning (IL)**, but IL is limited by the inconsistency and narrow coverage of expert data. **Offline RL** can extract better policies but suffers value overestimation on small datasets. **Online RL** on real robots is costly and unsafe; **simulator RL** suffers the sim-to-real gap. Meanwhile, diffusion models have become excellent real-world visual simulators. The paper's question: can a **diffusion world model** act as a learnable, high-fidelity simulator so that a pretrained policy can be refined by RL without touching the real robot?

The framing contrasts with prior world-model work in two ways: (1) IRASim and NWM (and V-JEPA 2) use generative/latent world models for **test-time planning**, not direct policy training; (2) **DiWA** does use a world model for policy learning but relies on **RSSM** (a VAE-based recurrent state-space model), whose blurry generations and compounding rollout errors limit performance. World4RL argues a diffusion backbone fixes the fidelity problem and thus enables stable **end-to-end** PPO refinement in imagination.

## Method

### Problem formulation

Given a BC-initialized stochastic policy, improve it via RL inside a learned world model. Three learned components:
- **Diffusion transition model** $D_\theta$: dynamics approximator, predicts next observation from history.
- **Reward classifier** $C_\psi$: binary success probability on the next observation (robotic manipulation has sparse rewards).
- **RL-refined policy** $\pi_\xi$: BC-initialized, refined with PPO; value network $V_\phi$.

### Core idea

Pretrain a diffusion world model that captures diverse dynamics on multi-task data, then **freeze it and refine the policy entirely within imagined rollouts**, avoiding real-world/simulator interaction. High generation fidelity is the enabler — a sharp, temporally consistent simulator makes imagined-rollout PPO stable.

### Architecture / algorithm

**Policy (BC pretraining).** Stochastic Gaussian policy conditioned on observation:

$$\pi_\xi(a_t \mid x_t) = \mathcal{N}(\mu_\xi(x_t), \Sigma_\xi(x_t)) \tag{1}$$

trained by maximizing expert log-likelihood (negative-log-likelihood BC loss):

$$\mathcal{L}_{BC}(\xi) = -\mathbb{E}_{(x_t, a_t)\sim D_{exp}}[\log \pi_\xi(a_t \mid x_t)] \tag{2}$$

**Reward classifier.** Given next observation $x_{t+1}$, $C_\psi$ outputs success probability $r := C_\psi(x_{t+1})$. It is trained not only on expert demos $D_{exp}$ but also on pre-trained-policy rollouts $D_{rollout}$ (so it can distinguish success states actually reached by the learned policy). Visual backbone = **pretrained ResNet18**, optimized with binary cross-entropy:

$$\mathcal{L}_C(\psi) = -\frac{1}{N}\sum_{i=1}^N [r_i \log C_\psi(x_i) + (1-r_i)\log(1-C_\psi(x_i))] \tag{3}$$

**Diffusion transition model (EDM preconditioning).** Following EDM, the denoiser uses the preconditioned form:

$$D_\theta(x^\tau; \tau, c) = c_{\text{skip}}^\tau x^\tau + c_{\text{out}}^\tau F_\theta(c_{\text{in}}^\tau x^\tau; c_{\text{noise}}^\tau, c) \tag{4}$$

where $F_\theta$ is the learnable network (U-Net 2D), $\tau$ the diffusion timestep, $c$ the conditioning. World dynamics are modeled by predicting the next clean observation $x_{t+1}^0$ from a history of observations $x_{t-T:t}^0$ and encoded actions $z_{t-T:t}$:

$$D_\theta(x^\tau, x_{t-T:t}^0, z_{t-T:t}) = c_{\text{skip}}^\tau x^\tau + c_{\text{out}}^\tau F_\theta(c_{\text{in}}^\tau x^\tau; c_{\text{noise}}^\tau, x_{t-T:t}^0, z_{t-T:t}) \tag{6}$$

with training objective (denoising-score-matching, EDM parametrization):

$$\mathcal{L}_D(\theta) = \mathbb{E}_{x^\tau \sim p_\tau}\left[\left\| F_\theta(c_{\text{in}}^\tau x_{t+1}^\tau, \tau, x_{t-T:t}^0, z_{t-T:t}) - \frac{1}{c_{\text{out}}^\tau}(x_{t+1}^0 - c_{\text{skip}}^\tau x_{t+1}^\tau)\right\|^2\right] \tag{7}$$

**Two-hot action encoding.** To feed continuous actions into the world model losslessly and differentiably (vs one-hot discretization, VQ-VAE latents, or FAST tokenization, which all introduce reconstruction error), each action dimension $a_i \in \mathbb{R}$ is mapped onto its two nearest bins $b_k \le a_i \le b_{k+1}$ from bin set $B = \{b_1,\dots,b_K\}$:

$$\mathbf{t}_i[k] = \frac{b_{k+1} - a_i}{b_{k+1} - b_k}, \qquad \mathbf{t}_i[k+1] = \frac{a_i - b_k}{b_{k+1} - b_k} \tag{5}$$

with $\sum_j \mathbf{t}_i[j] = 1$. This linear-interpolation weight vector $\mathbf{t}_i \in \mathbb{R}^K$ preserves continuity while embedding lightweight discrete structure; they use **K = 21 bins**. The encoded action $z$ replaces the raw action as conditioning input. Inspired by DreamerV3.

**World-model training data mixture.** The transition model trains on three sources: expert demos $D_{exp}$, pretrained-policy rollouts $D_{rollout}$, and **random rollouts $D_{rand}$** (uniform action sampling). Expert data gives successful trajectories, policy rollouts cover states seen during execution, and random rollouts broaden state-action coverage so the model does not overfit and generalizes during imagined RL rollouts.

**Policy optimization (PPO on imagined rollouts).** Standard clipped PPO objective:

$$\mathcal{L}_P(\xi) = \mathbb{E}_t[\min(\rho_t(\xi) A_t, \text{clip}(\rho_t(\xi), 1-\epsilon, 1+\epsilon) A_t)] \tag{8}$$

with probability ratio $\rho_t(\xi) = \pi_\xi(a_t\mid x_t)/\pi_{\xi_{old}}(a_t\mid x_t)$, advantage $A_t$, clip range $\epsilon$; and value loss

$$\mathcal{L}_V(\phi) = \mathbb{E}_t[(V_\phi(x_t) - (r_t + \gamma V_\phi(x_{t+1})))^2] \tag{9}$$

The frozen $D_\theta$ generates imagined next states; $C_\psi$ scores them for sparse binary reward $R(s_t, a_t)\in\{0,1\}$.

**Controlled policy exploration (OOD mitigation).** Imagined RL hits OOD actions that destabilize optimization. Fix: **clip the policy's predicted std** to a tighter bound — where common PPO allows $\sigma \le e^2$, World4RL imposes $\sigma \le e^0$. This keeps sampled actions near the world model's training support, reducing OOD rollouts.

### Derivations / why it works

_No new theoretical derivation; the method composes standard EDM diffusion (Eqs. 4,6,7), two-hot interpolation (Eq. 5), and clipped PPO (Eqs. 8,9)._ The load-bearing empirical argument is that diffusion fidelity + broad action-space coverage (random rollouts) + constrained exploration together keep imagined rollouts inside the world model's reliable support, which is what makes imagination-only PPO stable.

### Training procedure

Meta-World: per task, **50 expert + 150 pretrained-policy + 30 random** trajectories (230 total), 50 timesteps each; world model conditioned on **4 consecutive frames** + actions to predict the next; at test time it autoregressively generates from the initial frame+action. World4RL has **330M parameters** (comparable to NWM 320M, smaller than iVideoGPT 430M; DiWA is 40M). Hardware: **NVIDIA A800 (40GB)** — world-model pretraining ~20 h on 4 A800s; single-task policy refinement ~6 h on 1 A800.

### Inference / sampling

At deployment the refined policy $\pi_\xi$ runs directly on the robot (the world model is only used during training). During RL, the loop is: sample $a_t \sim \pi_\xi(\cdot\mid x_t)$ → two-hot encode → $D_\theta$ predicts $\tilde{x}_{t+1}$ → $C_\psi$ gives binary reward → PPO update.

## Experimental setup

- **World-model fidelity metrics:** LPIPS (perceptual), FID (image distribution quality), FVD (video-level spatio-temporal consistency). Baselines: NWM (DiT dynamics model), iVideoGPT (autoregressive transformer w/ compressive tokenization), DiWA (RSSM), plus DiWA-ST (single-task).
- **Policy metric:** task success rate (SR). Benchmark: **Meta-World**, 6 tasks. Baselines span IL (BC, Diffusion Policy), offline RL (TD3+BC, IQL), and world-model methods (IRASim, DiWA, TD-MPC2 — adapted to the same offline, no-online-interaction setting). Sparse success rewards used (not Meta-World's dense rewards).
- **Sample-efficiency comparison:** offline-to-online RLPD and Uni-O4 (which do use online interaction).
- **Real robot:** Franka Emika Panda, 6 manipulation tasks, HIL-SERL data protocol (space-mouse teleop): 50 expert + 50 pretrained-policy + 50 random trajectories per task; 20 physical rollouts per task with fixed initial scene/pose.

## Key results

- **Video-prediction fidelity (Table I):** World4RL gets the **lowest FVD/FID/LPIPS** under both policy and random rollouts — FVD 326.5 (policy) / 400.1 (random) vs NWM 547.4/851.9, iVideoGPT 450.3/531.3, DiWA 803.6/1231.0. DiWA collapses on the multi-task setting; DiWA-ST (644.8) still trails. Gains attributed to architecture (temporal consistency, visual fidelity), not scale.
- **Policy learning (Table II, Meta-World, 3 seeds):** World4RL avg **67.5%** SR vs BC 51.5, DP 51.5*, TD3+BC 45.0, IQL 57.7, IRASim 42.0, DiWA 57.0, TD-MPC2 60.0 — i.e. **+16 over BC base** and best overall. Largest gains on harder tasks (coffee-pull +21, soccer +13, lever-pull +21). (*DP average not separately reconstructed here.)
- **Sample efficiency (Fig. 3):** World4RL matches/exceeds RLPD and Uni-O4 using **zero online samples**, whereas RLPD needs ~**346k** and Uni-O4 ~**470k** online steps to reach the same level.
- **Real Franka (Table III, 20 trials/task):** World4RL avg **93.3%** vs BC 68.3% (**+25**) and Diffusion Policy 88.3%. Perfect on Pick-bread-out (20/20) and Close-drawer (20/20). Qualitatively, refined policies act more **decisively** (less hesitation/lingering than BC/DP).

## Ablations

- **Action encoding (Table IV):** two-hot beats one-hot, linear projection, FAST tokenization, and VQ-VAE across FVD/FID/LPIPS (two-hot FVD 326.5 policy / 400.1 random; VQ-VAE worst at 525.6/860.0). Two-hot preserves continuity while adding lightweight discreteness, avoiding tokenization/latent reconstruction errors.
- **Policy-optimization design (Fig. 5):** removing either **action-std clipping** or **random rollouts** degrades performance on Door-lock and Lever-pull; w/o std clipping also shows much higher training variance — controlled exploration matters for both final performance and stability. Random rollouts (broad action coverage) matter for reliable imagined transitions.

## Limitations

Paper's own: (1) The work "establishes feasibility rather than exhausting potential"; (2) **moderate visual resolution and model capacity** due to compute limits, which caps imagined-rollout fidelity; future work aims at richer visual representations and RL algorithms robust to imperfect learned dynamics.

Honest reader's view [analyst's view]: (a) Only **6 Meta-World tasks** and 6 real tasks, all fairly short-horizon (50 timesteps) — no long-horizon or high-DOF evaluation. (b) The reward classifier is a **binary success detector**; performance hinges on its accuracy, and a policy could exploit classifier errors ("reward hacking" against a learned reward inside a learned model) — not stress-tested. (c) Refinement is **single-task** (one world model + policy refined per task); scalability to many-task/generalist policies is unaddressed. (d) Gains over the strongest baselines (TD-MPC2 60.0, IQL 57.7) are real but modest (~7–10 points) relative to the headline "+16 over BC." (e) The frozen world model can't correct its own compounding errors during long imagined rollouts — mitigated by std clipping but not eliminated.

## Why it matters [analyst's view]

World4RL is a clean instance of the **"train RL inside a generative world model"** recipe specialized to manipulation, and its most transferable contributions are practical: (1) **two-hot action encoding** as a lossless, differentiable interface between a continuous-action RL policy and a diffusion dynamics model — a concrete answer to "how do you condition an image world model on continuous robot actions" that beats tokenization/VQ; (2) the **random-rollout + std-clipping** pair as a recipe for keeping imagination-only PPO inside the world model's support, addressing the central failure mode of model-based RL (policy drifts into regions where the model is wrong). It also provides direct evidence that **diffusion beats RSSM** for manipulation world models (the DiWA comparison), which is a useful data point for the broader latent-vs-pixel world-model debate. Its sample-efficiency story (matching 300k+-online-step methods with zero online steps) is the strongest practical selling point for real-robot deployment.

## Open questions / things to verify

- How robust is the frozen world model to reward-hacking by the PPO policy over many refinement epochs?
- Does the approach scale beyond single-task refinement to multi-task/generalist policies without retraining a world model per task?
- How does fidelity/performance degrade at longer horizons than 50 steps, and how sensitive is everything to the reward classifier's error rate?
- Why exactly $\sigma \le e^0$ — how sensitive is stability to this exploration bound?
- Comparison to WAM-style approaches (e.g. joint video-action models) that don't separate transition model, reward classifier, and policy.

## Connections

- Topic MOCs: [[topics/world-models]], [[topics/diffusion-models]], [[topics/reinforcement-learning]], [[topics/robotics]]
- Author indices: [[authors/zhennan-jiang]], [[authors/haoran-li]], [[authors/dongbin-zhao]], [[authors/chao-yu]]
- Related: [[papers/porcher-2026-flowwm]] (generative world model), [[papers/ye-2026-world-action-models]] (contrast: pixel-space WAM policy vs diffusion world model for RL refinement)
- Contrasts with: DiWA (RSSM world model), IRASim/NWM (world models for planning not policy training), TD-MPC2 (latent model-based RL)

## Selected quotes

> "World4RL, a framework that employs diffusion-based world models as high-fidelity simulators to refine pre-trained policies entirely in imagined environments for robotic manipulation." — Abstract

> "Unlike one-hot discretization, latent-space representation, or token-based approaches, two-hot encoding provides a lossless and differentiable representation without introducing reconstruction errors." — §III-B

> "While a common PPO implementation allows σ ≤ e², we instead impose a much tighter constraint σ ≤ e⁰. This constrained exploration keeps sampled actions closer to the support of the world-model training distribution." — §III-C
