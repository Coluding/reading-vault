---
type: paper
title: "AdaWorld: Learning Adaptable World Models with Latent Actions"
authors: ["Shenyuan Gao", "Siyuan Zhou", "Yilun Du", "Jun Zhang", "Chuang Gan"]
year: 2025
venue: "ICML 2025"
url: https://arxiv.org/abs/2503.18938
rw_id: 01kx6ckh5bvv4gx1wst0tzx8zp
topics: [world-models, video-generation, self-supervised-learning, imitation-learning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-14
last_updated: 2026-07-14
---

## TL;DR

AdaWorld is a world-model *pretraining* recipe that bakes action information into the model without any action labels, so the model can adapt to new environments from a handful of interactions. The trick: a latent-action autoencoder extracts a compact, **continuous** latent action $\tilde{a}$ from every pair of consecutive video frames via an information bottleneck ($\beta$-VAE), capturing "what changed" while discarding context. An autoregressive, frame-level diffusion world model (built on Stable Video Diffusion) is then pretrained conditioned on these latent actions across ~2 billion frames from 1000+ environments. Because the latent actions are context-invariant, AdaWorld can (1) transfer a demonstrated action to a new scene *with no training*, and (2) adapt to a new environment's real action space by initializing the control interface from averaged latent actions and finetuning briefly. It beats action-agnostic pretraining and flow/discrete-action-conditioned baselines on action transfer (FVD 767 vs 1545 on LIBERO), simulation quality after adaptation, and visual planning — reaching 56.7% avg success on Procgen goal-reaching (vs 26% action-agnostic) and even 44.8% with *no* finetuning. Published at ICML 2025.

## Context & motivation

World models learn action-controlled future prediction and are increasingly initialized from pretrained video generators. The problem: to acquire precise action controllability, existing models need substantial **action-labeled** data and costly training, and adapting them to a *new* environment with a *different* action specification requires expensive re-training. Pseudo-labeling videos is possible (e.g. VPT), but there is no unified action format across heterogeneous environments (a game's button presses vs a robot's joint displacements vs ego-motion). The paper's premise, motivated by how humans reuse internal action representations learned from observation, is: *incorporate action information during pretraining* — but do it from unlabeled video via a learned, transferable latent action space, so adaptation to a new environment reduces to *finding the mapping* between that environment's actions and the pretrained latent action interface. This contrasts with prior latent-action work (Genie, IGOR, Moto) that targeted playability/behavior-cloning and used *discrete* latent actions; AdaWorld argues a *continuous* latent action space is more expressive and composable.

## Method

AdaWorld has two components: (1) a **latent action autoencoder** that extracts actions from unlabeled video, and (2) an **autoregressive diffusion world model** conditioned on those latent actions.

### Problem formulation

Given unlabeled videos (no action labels), learn a compact latent action $\tilde{a}$ representing the transition between consecutive frames $f_{t:t+1}$, and a world model that predicts $f_{t+1}$ from history plus $\tilde{a}$. Objective: maximize adaptability — the ability to transfer/learn new actions in unseen environments from limited interactions and finetuning.

### Core idea

Actions drive the *dominant variation* in interactive scenarios. Force a tight information bottleneck on a latent that must explain the next frame given the current one, and it will disentangle the agent's action from static context — yielding a context-invariant, transferable action code. Then pretrain a world model conditioned on that code; adapting to a new environment becomes fitting its action space onto the already-learned latent action interface.

### Architecture / algorithm

**Latent action autoencoder (Transformer-based, information-bottleneck).** The encoder takes two consecutive frames $f_{t:t+1}$, splits each into $16\times16$ patches, projects to patch embeddings, flattens spatially, and concatenates two learnable tokens $a_{t:t+1}$. Sinusoidal position embeddings mark spatial location. A **spatiotemporal Transformer** with $L$ stacked blocks (interleaved spatial + temporal attention, then FFN — the Genie-style factorization) processes these: spatial attention attends over all tokens within a frame; temporal attention connects tokens at the same spatial position across the two frames, with rotary embeddings encoding the causal (temporal) order. After enough attention, the learnable token $a_{t+1}$ aggregates the inter-frame dynamics; all other tokens are discarded, and $a_{t+1}$ is projected to the posterior parameters $(\mu_{\tilde{a}}, \sigma_{\tilde{a}})$ of a **VAE** (Kingma & Welling). Sample $\tilde{a}$ from the posterior, attach it to $f_t$, and feed to the decoder.

The **latent action decoder** is a *spatial* Transformer that predicts the next frame $f_{t+1}$ in pixel space from $\tilde{a}$ and $f_t$. Training objective is the (β-)VAE ELBO. The plain-VAE form (Eq. 1):

$$\mathcal{L}^{pred}_{\theta,\phi}(f_{t+1}) = \mathbb{E}_{q_\phi(\tilde{a}\mid f_{t:t+1})}\log p_\theta(f_{t+1}\mid \tilde{a}, f_t) - D_{KL}\!\left(q_\phi(\tilde{a}\mid f_{t:t+1}) \,\|\, p(\tilde{a})\right)$$

where $q_\phi$ is the encoder (posterior over latent action), $p_\theta$ the decoder (frame likelihood), and $p(\tilde{a})$ the prior. The reconstruction term forces $\tilde{a}$ to carry the information needed to predict $f_{t+1}$; the KL term compresses $\tilde{a}$. Because $\tilde{a}$ is extremely low-dimensional relative to a full frame, it *cannot* smuggle the whole next frame through — it must encode only the *critical variation* (the action), giving context-invariance.

Empirically the standard VAE's KL constraint is too strong and the latent action fails to express diverse transitions; dropping it hurts disentanglement. They adopt the **$\beta$-VAE** (Eq. 2) with a tunable $\beta$ weighting the KL:

$$\mathcal{L}^{pred}_{\theta,\phi}(f_{t+1}) = \mathbb{E}_{q_\phi(\tilde{a}\mid f_{t:t+1})}\log p_\theta(f_{t+1}\mid \tilde{a}, f_t) - \beta\, D_{KL}\!\left(q_\phi(\tilde{a}\mid f_{t:t+1}) \,\|\, p(\tilde{a})\right)$$

$\beta$ trades expressiveness (low $\beta$ → more information in $\tilde{a}$, more differentiable) against context-disentanglement (high $\beta$ → actions from different environments overlap/cluster). Default $\beta = 2\times10^{-4}$.

**Autoregressive diffusion world model.** Rather than repurposing the coarse single-pass latent-action decoder (which degrades badly over rollouts), they build a separate world model from **Stable Video Diffusion (SVD)** — a latent diffusion model trained in the **EDM** framework (Karras et al. 2022). Key modifications:
- **Frame-level control**: unlike clip-predicting video models, AdaWorld denoises **one** frame at a time, giving finer interaction granularity.
- **Latent action conditioning**: the latent action $\tilde{a}$ is concatenated with *both* the timestep embedding and SVD's CLIP image embedding, for deep fusion with the action signal.
- **Short-term memory of $K$ history frames**: the last frame in memory is SVD's condition image; historical frames are encoded with the SVD image encoder and concatenated with the noise latent of the target frame, inheriting SVD's temporal modeling. History length is randomly sampled up to a max of 6 during training, with the memory length passed as a condition.
- **Noise augmentation** on historical frames during training (per He et al. 2022; Valevski et al. 2025) mitigates long-term drift, even when no noise is added at inference.

Pretraining loss (Eq. 3), an EDM-style $x_0$-prediction denoising objective:

$$\mathcal{L}_{\text{pretrain}} = \mathbb{E}_{x_0,\epsilon,t}\left[\, \| x_0 - \hat{x}_0(x_t, t, c) \|^2 \,\right]$$

where $x_0$ is the clean target-frame latent, $x_t$ its noised version at diffusion timestep $t$, $\hat{x}_0$ the model's prediction, and $c$ the conditioning = history frames + latent action $\tilde{a}$.

**Adaptation mechanisms (Sec 2.3):**
- *Action transfer (no training)*: encode a demo video's frames into a latent-action sequence, then reuse that sequence as conditions to autoregressively roll out from a new initial frame in a different context.
- *World model adaptation (discrete actions)*: collect a few action-video pairs, infer their latent actions, and — exploiting continuity of the space — **average** the latent actions per label. For $N$ discrete actions, initialize a specialized control interface with $N$ averaged latent actions and finetune the whole model briefly.
- *Continuous action spaces*: add a lightweight **MLP** mapping raw action inputs to the latent action interface, initialized by finetuning the MLP on a few action↔latent-action pairs.
- *Action composition / creation*: averaging two latent actions in the continuous space yields a new action merging both functions (Fig 5); clustering collected latent actions creates a flexible set of control options — positioning AdaWorld as an alternative to generative interactive environments like Genie.

### Derivations / why it works

The load-bearing argument is the information-bottleneck reasoning around Eqs. 1–2 rather than a formal theorem. The $\beta$-VAE KL penalty $\beta D_{KL}(q_\phi\|p)$ upper-bounds the information the latent action can carry; minimizing reconstruction error *subject to* that bound forces $\tilde{a}$ to allocate its limited capacity to the highest-variance, most predictive factor — the agent's action — while context (colors, textures, object arrangement) is supplied by the conditioning frame $f_t$ and need not be encoded. The result is a latent that is (empirically) context-invariant: UMAP (Fig 7) shows the same action across *different* environments clustering together. Continuity of the space is what makes label-averaging and MLP-mapping valid initializations. No convergence/optimality proof is given; the justification is empirical.

### Training procedure

- **Dataset**: four public datasets — Something-Something v2 (Goyal 2017), Ego4D (Grauman 2022), Open X-Embodiment (O'Neill 2024), MiraData (Ju 2024) — plus videos auto-collected from **1016 environments** across Gym Retro (Nichol 2018) and Procgen (Cobbe 2020). Total ≈ **2000 million frames** of interactive scenarios spanning ego and third-person views, virtual games, and real-world activity.
- **World model** initialized from SVD (EDM). Baselines and AdaWorld all trained **50K iterations** for fair comparison.
- **Adaptation (simulation)**: per unseen environment, only **100 samples per discrete action** (or 100 trajectories for nuScenes); finetune **800 steps**, batch size **32**, lr **5×10⁻⁵**, with pretrained weights' lr discounted ×0.1. The nuScenes continuous-action MLP is finetuned **3K steps** in <30 s on a single GPU.
- **Planning (Procgen)**: 100 samples per action (LEFT/DOWN/UP/RIGHT), finetune 500 steps, same batch/lr.
- Latent-action data-diversity ablation autoencoders trained 40K steps.

### Inference / sampling

Autoregressive frame-by-frame rollout: predict $f_{t+1}$ via diffusion denoising conditioned on the latent action and the memory of $K$ history frames, append the prediction to memory, repeat. Frame-level granularity. For planning, the world model is used inside sampling-based MPC — Cross-Entropy Method for Procgen games, MPPI (model-predictive path integral) for VP2 robot tasks — with reward = cosine similarity between predicted observation and goal image.

## Experimental setup

- **Baselines**: (1) *action-agnostic* pretraining (same architecture, always fed zero action condition); (2) *optical-flow-conditioned* (UniMatch flows downsampled to 16×16 as the action condition); (3) *discrete latent action* (VQ-VAE variant with an 8-code codebook, à la Genie).
- **Action transfer** eval: unseen LIBERO and SSv2, 1300 paired videos; metrics FVD↓, Embedding Cosine Similarity (ECS, I3D-based) ↑, plus human judgment on 50 pairs each.
- **Simulation quality**: unseen Habitat, Minecraft, DMLab (discrete) and nuScenes (continuous), 300 val samples each; PSNR↑, LPIPS↓.
- **Planning**: Procgen goal-reaching (Heist, Jumper, Maze, CaveFlyer, 30 scenes each) via MPC; VP2 robot benchmark (Robosuite + RoboDesk) via MPPI.
- **Generality**: apply action-aware pretraining to iVideoGPT; evaluate on BAIR pushing.

## Key results

- **Action transfer (Table 1)**: AdaWorld FVD **767.0** / ECS **0.804** / Human **70.5%** on LIBERO vs best baseline 1409.5 / 0.724 / 3.5%; on SSv2 FVD **473.4** / ECS 0.639 / Human 61.5%. Large, decisive margin.
- **Simulation after adaptation (Table 2)**: best PSNR/LPIPS on all four unseen environments (e.g. Habitat PSNR 23.58 / LPIPS 0.327; nuScenes 21.60 / 0.436). All action-aware variants beat action-agnostic, confirming the central claim.
- **Planning in games (Table 3)**: average success **56.7%** with finetuning, **44.8%** without any finetuning (just averaged latent actions), vs **26.0%** action-agnostic, 26.2% random, 27.2% Q-learning; Oracle (GT simulator MPC) 80.7%. Notably the no-finetune variant beats the *finetuned* action-agnostic baseline.
- **Robot planning VP2 (Table 4)**: aggregate normalized success **21.54** vs **5.03** for action-agnostic (e.g. Robosuite push 63.5% vs 17.5%).
- **Generality (Table 6)**: iVideoGPT+AdaWorld improves BAIR PSNR 16.59→17.40, LPIPS 0.220→0.204 — the recipe ports to other world models.

## Ablations

- **Interface initialization (Fig 6)**: random-init control interface starts *below* action-agnostic but surpasses it after ~200 finetuning steps; latent-action init is best throughout — the pretrained interface is the source of sample/finetune efficiency.
- **Data diversity (Table 5)**: Retro+OpenX gives best Procgen latent-action generalization (PSNR 26.62 / LPIPS 0.234); surprisingly, adding real-world robot video (OpenX) *helps* generalization to unseen 2D games — more data diversity helps.
- **$\beta$ choice (Fig 7, UMAP)**: same actions across environments cluster together, validating context-invariance; lower $\beta$ → more differentiable but less cross-environment overlap (worse disentanglement). Default $\beta = 2\times10^{-4}$.
- **Composition (Fig 5)**: averaging two latent actions yields an action merging both, evidencing a semantically continuous space.

## Limitations

Paper's own: (1) not real-time — inference is slow (suggests distillation/few-step sampling as future work); (2) like prior video world models, struggles to *create novel content* once a rollout leaves the initial scene (expected to improve with model/data scale); (3) falls short at extremely long-term rollouts. [analyst's view] Additional flags: the "action" disentangled by the bottleneck is whatever dominates frame-to-frame variation, so in scenes with large camera motion or dynamic non-agent objects the latent may capture the wrong factor; UMAP shows real noise from actions that can't execute in certain states; and evaluation leans on FVD/PSNR/LPIPS + a small (50-pair, 4-volunteer) human study, which is thin for the strong "action successfully transferred" claim.

## Why it matters [analyst's view]

AdaWorld reframes world-model adaptation as an *interface-alignment* problem: pretrain once on a universal continuous action code, then map any new environment's action space onto it cheaply. That is an attractive alternative to the dominant "scale action-labeled data per environment" path and connects three threads — self-supervised action/representation learning, video-diffusion world models, and model-based planning/control. The continuous-vs-discrete latent action choice (vs Genie) is the substantive bet, and the composition/creation results suggest the space has genuine algebraic structure worth exploiting. For robotics and embodied RL it offers a concrete route to few-shot controllable simulators for MPC. The "action-aware pretraining ports to iVideoGPT" result is the most portable takeaway: it's a *recipe*, not just a model.

## Open questions / things to verify

- How sensitive is disentanglement to $\beta$ and to scenes where the agent is *not* the dominant motion source? What happens with multi-agent or heavy background dynamics?
- The averaged-latent-action initialization assumes intra-label latent actions are tight clusters — how does variance scale with action-space size / continuous actions?
- Real-time and long-horizon rollout are open; how much do the cited distillation/diffusion-forcing fixes actually recover?
- Human eval is small; a larger controlled study of "action transfer success" would strengthen Table 1.

## Connections

- Builds on: Stable Video Diffusion + EDM diffusion; Genie's spatiotemporal Transformer / discrete latent actions (contrasted); $\beta$-VAE information bottleneck. _needs note_ for Genie (Bruce et al. 2024), iVideoGPT (Wu et al. 2024), IGOR, Moto.
- Related world models in vault: [[papers/ding-2024-diffusion-world-model]] (diffusion world model for planning), [[papers/jiang-2025-world4rl]] (world model for RL), [[papers/ye-2026-world-action-models]] (joint world+action modeling), [[papers/maes-2026-leworldmodel]] (learning world models).
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/self-supervised-learning]], [[topics/imitation-learning]]
- Author indices: [[authors/yilun-du]]

## Selected quotes

> "The key idea is to incorporate action information during the pretraining of world models. This is achieved by extracting latent actions from videos in a self-supervised manner, capturing the most critical transitions between frames." — Abstract

> "adapting to a new environment is akin to finding the mapping of corresponding latent actions for its action space." — §1

> "we compress the latent actions into a continuous latent space to maximize expressiveness and enable flexible composition." — §1
