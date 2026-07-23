---
type: paper
title: "Sensorimotor World Models: Perception for Action via Inverse Dynamics"
authors: ["Petr Ivashkov", "Randall Balestriero", "Bernhard Schölkopf"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2606.20104
rw_id: 01ky7gh1atgh8rrzwtm762jeae
topics: [world-models, jepa, self-supervised-learning, representation-learning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# Sensorimotor World Models: Perception for Action via Inverse Dynamics

## TL;DR

SMWM (Sensorimotor World Model) is a JEPA-style latent world model trained end-to-end from pixels whose **sole anti-collapse mechanism is a single-step inverse-dynamics head**: a small MLP that predicts the executed action from consecutive latent states, with gradients flowing back into the encoder. The joint objective $\mathcal{L} = \mathcal{L}_{\text{fwd}} + \lambda \mathcal{L}_{\text{inv}}$ adds exactly one hyperparameter and needs no frozen pretrained encoder, no EMA/stop-gradient targets, and no distributional regularizer — the mechanisms used by DINO-WM, V-JEPA 2, and LeWorldModel/SIGReg respectively. The inverse loss simultaneously prevents collapse and biases the encoder toward the **controllable degrees of freedom** of the environment: on controlled dot-world testbeds the latent PCA rank exactly matches the number of controllable dimensions, uncontrollable distractor objects are filtered out, and actions act approximately as latent translations. On four offline reward-free control tasks, CEM/MPC planning in the learned latent space matches the SIGReg baseline on the three 2D tasks and clearly beats it on 3D OGBench-Cube (84% vs 59% success). The paper frames this as an operational instantiation of "perception for action" / sensorimotor-contingency ideas from cognitive science.

## Context & motivation

Latent world models sidestep pixel-space prediction by learning dynamics in a compact embedding space, and JEPAs push this to its limit by dispensing with reconstruction entirely and predicting directly in embedding space. The catch is **representation collapse**: if encoder and latent dynamics model are trained jointly only to make embedding-space prediction easy, the encoder can map every observation to a constant embedding, making the prediction task trivially solvable and the model useless. The paper surveys how recent JEPA world models each buy stability differently: DINO-WM freezes a pretrained DINOv2 encoder; PLDM adds a multi-term variance–covariance (VICReg-style) regularizer plus auxiliary terms; LeWorldModel matches the embedding distribution to an isotropic Gaussian via SIGReg; V-JEPA 2 uses stop-gradient targets from an EMA target encoder.

The authors connect the collapse problem to a more basic question — *what should a world-model representation retain?* — and answer it via "perception for action" from cognitive psychology/neuroscience (Goodale & Milner), common coding (Prinz), Gibson's affordances, von Uexküll's Umwelt, enactive perception (Varela et al.), and sensorimotor contingencies (O'Regan & Noë): representations should preserve the distinctions that matter for the agent's actions and their consequences, not visual fidelity per se. They also link it to causal representation learning: a causal representation should encode actions and their effects, providing an interventional world model. The setting is offline datasets of video frames with continuous action labels — no rewards, no task labels, no knowledge of the data-collecting policy (covers demonstrations, action-annotated video, logged interaction data). Contributions: (1) SMWM with inverse-dynamics regularization as the sole anti-collapse mechanism and one extra hyperparameter; (2) evidence the learned representations track controllable degrees of freedom, preserve spatial geometry, and filter uncontrollable distractors; (3) competitive planning against SIGReg baselines on 2D and 3D control.

## Method

### Problem formulation

Given an offline dataset $\mathcal{D} = \{(o_t, a_t, o_{t+1})\}$ of transitions — $o_t, o_{t+1} \in \mathcal{O}$ consecutive observations (video frames), $a_t \in \mathcal{A} \subseteq \mathbb{R}^m$ a continuous action taken in between — learn an encoder $f_\theta : \mathcal{O} \to \mathcal{Z}$ mapping observations into a compact embedding space $\mathcal{Z} \subseteq \mathbb{R}^d$, together with a forward dynamics model $g_\phi : \mathcal{Z} \times \mathcal{A} \to \mathcal{Z}$ predicting the next embedding from the current embedding and action. No behavior policy or reward function is required.

### Core idea

If two consecutive embeddings contain enough information to recover the action that produced the transition, they must preserve the observation features relevant to controllable dynamics. So instead of prescribing embedding-space geometry (as distributional regularizers do), anchor the representation to a **task-grounded quantity — the action** — via an inverse-dynamics prediction head whose gradients flow into the encoder.

### Architecture / algorithm

**Forward dynamics in embedding space.** Both observations are encoded and the dynamics model is trained to match the next embedding:

$$z_t = f_\theta(o_t), \quad z_{t+1} = f_\theta(o_{t+1}), \quad \hat{z}_{t+1} = g_\phi(z_t, a_t)$$

with the mean-squared forward loss

$$\mathcal{L}_{\text{fwd}} = \mathbb{E}_{(o_t, a_t, o_{t+1}) \sim \mathcal{D}} \left[ \| \hat{z}_{t+1} - z_{t+1} \|_2^2 \right].$$

Here $z_t, z_{t+1} \in \mathcal{Z}$ are the encoded current/next observation and $\hat{z}_{t+1}$ is the predicted next embedding. This objective alone admits a trivial global minimum: if $f_\theta$ maps every observation to a constant $z^\star$ and $g_\phi$ outputs $z^\star$ regardless of input, $\mathcal{L}_{\text{fwd}} = 0$ — a fully collapsed, useless representation. This is the failure the regularizer must block.

**Inverse dynamics regularization.** An inverse model $h_\psi : \mathcal{Z} \times \mathcal{Z} \to \mathcal{A}$ predicts the executed action from the two embeddings, $\hat{a}_t = h_\psi(z_t, z_{t+1})$, trained with the mean-squared action loss

$$\mathcal{L}_{\text{inv}} = \mathbb{E}_{(o_t, a_t, o_{t+1}) \sim \mathcal{D}} \left[ \| \hat{a}_t - a_t \|_2^2 \right].$$

*Why this prevents collapse:* under squared loss, a collapsed encoder feeds the inverse model identical inputs for every transition, so the best possible inverse predictor is the constant action $\mathbb{E}[a_t]$, with risk $\mathbb{E}\|a_t - \mathbb{E}[a_t]\|_2^2$. Any reduction below this constant-predictor risk requires $(z_t, z_{t+1})$ to carry information predictive of $a_t$ — a nontrivial inverse prediction therefore rules out full collapse. Crucially, unlike SIGReg's isotropic-Gaussian prior, this imposes **no geometry** on $\mathcal{Z}$; it only requires the encoder to preserve whatever information about $a_t$ the pair $(o_t, o_{t+1})$ contains.

**Joint objective.** Encoder, forward, and inverse models are optimized jointly:

$$\mathcal{L} = \mathcal{L}_{\text{fwd}} + \lambda \mathcal{L}_{\text{inv}}, \qquad \lambda > 0,$$

where $\lambda$ weights the inverse regularization. $\mathcal{L}_{\text{fwd}}$ updates $f_\theta$ and $g_\phi$; $\mathcal{L}_{\text{inv}}$ updates $f_\theta$ and $h_\psi$. The encoder thus receives gradients from both, producing embeddings simultaneously predictive under forward dynamics and action-informative under inverse dynamics. (The paper's Alg. 1 is literally ~8 lines of PyTorch: two encodings, one forward MSE, one inverse MSE, weighted sum; default `lambda_inv=10.0` in the toy setting.) A key architectural contrast with ICM (Pathak et al. 2017), the canonical inverse-dynamics feature learner: in ICM the encoder receives gradients *only* from the inverse loss (the forward loss stop-gradients the target and is used only for intrinsic reward); here the encoder is trained through both losses because the forward predictor *is* the deliverable world model.

### Derivations / why it works

**Equivariance from the forward loss.** Writing $a(o)$ for the effect of action $a$ on observation $o$, and $g_a(z) := g_\phi(z, a)$, the forward loss minimizes $\|f(a(o)) - g_a(f(o))\|_2^2$, encouraging the consistency condition

$$f(a(o)) \approx g_a(f(o))$$

on the training support — a commutative diagram: encoding after the physical intervention agrees with encoding first and then applying the learned latent intervention. This is an *equivariance* statement (the latent state transforms covariantly with the intervention).

**Theorem 1 (latent action composition).** If $\mathcal{A}$ is a semigroup under composition and $f$ is exactly equivariant ($f(a(o)) = g_a(f(o))$ for all $a, o$), then $a \mapsto g_a$ is a **homomorphism** on the encoded manifold $f(\mathcal{O})$: $g_{a_2 \circ a_1}(z) = g_{a_2}(g_{a_1}(z))$ for all $z \in f(\mathcal{O})$. Proof sketch (App. G): pick $o$ with $f(o) = z$; apply equivariance three times — $g_{a_2 \circ a_1}(z) = f(a_2 \circ a_1(o)) = f(a_2(a_1(o))) = g_{a_2}(f(a_1(o))) = g_{a_2}(g_{a_1}(z))$. Physical action composition must be represented by composition of latent interventions (approximately and on-support in the learned model). The authors relate this to MDP homomorphisms and causal abstraction, but note the homomorphism here *emerges* from the forward objective rather than being assumed.

**Why equivariance alone is not enough.** Two degenerate solutions satisfy exact equivariance: (1) *no encoding* ($\mathcal{Z} = \mathcal{O}$, $f = \mathrm{id}$, $g_a = a$ — no compression) and (2) *collapse* ($\mathcal{Z} = \{z\}$, $f \equiv z$, $g_a = \mathrm{id}$ — no information). Useful representations need **faithfulness** (every action acting nontrivially on $\mathcal{O}$ must act nontrivially on $f(\mathcal{O})$), and more strongly **latent action identifiability**: whenever $z' = g_a(z)$, the action $a$ is recoverable from $(z, z')$. The inverse model enforces exactly this constraint.

**Emergent translation structure.** Empirically the learned interventions act approximately as latent translations, $g_a(z) \approx z + \rho(a)$ with the latent effect vector $\rho(a)$ approximately independent of $z$. Substituting into the composition identity gives $\rho(a_2 \circ a_1) \approx \rho(a_1) + \rho(a_2)$ — an approximately *additive* representation of interventions, though nothing in the objective enforces additivity. The proposed explanation: a low-complexity way to make $a_t$ recoverable from $(z_t, z_{t+1})$ is to encode it in the displacement, $z_{t+1} - z_t \approx \rho(a_t)$; if $\rho$ is injective on the action support, $h$ decodes by approximately inverting $\rho$. The authors read this as an *interventional* version of the linear representation hypothesis: not merely that actions are linearly decodable, but that applying an intervention corresponds to moving along an action-dependent direction, $f(a(o)) - f(o) \approx \rho(a)$. (Caveat footnote: this need not hold for contact-rich dynamics, boundary effects, periodic variables, state-dependent action effects, or noncommutative action laws.)

### Training procedure

Two regimes. **Dot/sprite-world testbeds:** 64×64 RGB canvas; encoder = 3 stride-2 convs (3→32→64→128, 3×3 kernels) + ReLU, 2×2 max-pool, linear to $d = 64$, final tanh; $g_\phi, h_\psi$ = 2-hidden-layer MLPs, width 256 ($g_\phi$ ends in tanh, $h_\psi$ no output activation); ~220k/100k/100k params. Adam, lr 5e-4, no weight decay, 100 epochs, batch 256, 250k transitions, $\lambda = 10$, actions normalized by $\Delta_{\max}$ to $[-1,1]$.

**Control benchmarks:** LeWM's architecture essentially unchanged — encoder is ViT-Tiny (patch 14, ~5M params), final CLS token projected to $z \in \mathbb{R}^{192}$; forward model is a small ViT-S transformer (~10M params) with the action injected via AdaLN-zero, but with **history length 1 on every environment** (LeWM uses history 3 on Push-T and Cube). The only architectural addition is $h_\psi$: a 2-layer MLP of width 256 mapping $[z_t; z_{t+1}] \in \mathbb{R}^{2d} \to \hat{a}_t \in \mathbb{R}^m$ (~0.1M params). AdamW, lr 1e-4, weight decay 1e-3, batch 256, 10 epochs, grad clip 1.0; datasets loaded with frameskip 5, so the action encoder and inverse head receive the concatenated action over the skipped interval. $\lambda$ is environment-specific: TwoRoom 0.1, Reacher 5, Push-T 30, OGBench-Cube 1. SIGReg baseline uses identical optimizer/architecture with $\lambda = 0$ and SIGReg weight 0.09. Single H100; each world model trains in under 4 hours. Everything runs through the stable-worldmodel codebase.

### Inference / planning

Goal-conditioned trajectory optimization in frozen latent space, following LeWM's protocol: encode current and goal observations, $z_1 = f_\theta(o_1)$, $z_g = f_\theta(o_g)$; roll out $\hat{z}_{t+1} = g_\phi(\hat{z}_t, a_t)$ autoregressively over horizon $H$ for candidate action sequences; score each by the terminal goal-matching cost

$$\mathcal{C}(a_{1:H}) = \| \hat{z}_{H+1} - z_g \|_2^2,$$

minimized with the Cross-Entropy Method (sample Gaussian action sequences, score via latent rollouts, refit to top elites), wrapped in receding-horizon MPC (execute the first $K \le H$ actions, re-encode, replan) to limit compounding rollout error. Hyperparameters (shared across all four environments): $H = 5$ model steps, $K = 5$, each model-step action = a block of 5 primitive actions, CEM population 300, 30 iterations, 30 elites, 100 evaluation tasks, goal placed 25 primitive steps ahead, budget 50 primitive steps. $f_\theta, g_\phi$ stay frozen; only the action sequence is optimized.

## Experimental setup

- **Testbeds (latent-structure analysis):** *dot world* — colored dots on a 64×64 white canvas, actions = 2D displacements per controlled dot, ground-truth state known; four configurations varying controllable DOF (Independent: 2 free dots, $|a| = 4$; Coupled: pair sharing one displacement, $|a| = 2$; Distractor: 1 controlled + 1 randomly moving dot, $|a| = 2$; Combined: $|a| = 6$ + distractor). *Sprite world* — asymmetric triangle with pose $(x, y, \theta)$, action repertoire varied (no control / x-y control / x-y-θ control) while uncontrolled DOF keep moving randomly.
- **Control benchmarks:** TwoRoom (2D navigation, 10k noisy scripted rollouts), Reacher (DMC two-link arm, DINO-WM variant with joint-configuration success, 10k SAC episodes), Push-T (2D contact-rich manipulation, 20k DINO-WM expert demos), OGBench-Cube (3D tabletop single-cube manipulation, 10k scripted trajectories). 90/10 episode-level train/val split; all metrics on held-out episodes.
- **Baselines:** SIGReg (same encoder/predictor, Gaussian-matching regularizer instead of $\mathcal{L}_{\text{inv}}$ — the main head-to-head), Forward-only ($\lambda = 0$ ablation), Random actions (floor).
- **Metrics:** goal-conditioned planning success rate (mean ± SE over 5 seeds); held-out $R^2$ of linear (ridge, $\alpha = 10^{-3}$) and 2-layer MLP probes regressing ground-truth physical state from frozen embeddings (25k train / 5k val embeddings); PCA spectra and projections of held-out embeddings.

## Key results

- **Latent structure (dot world):** the PCA spectrum of embeddings drops sharply at the true intrinsic dimension ($d_{\text{true}} = 2$ inside $d = 64$ ambient); the state grid maps to a near-square, spatially faithful latent grid (rotation only — PC axes identifiable up to rotation/sign). Across all four DOF configurations the number of significant PCs **exactly matches the controllable dimension** (4, 2, 2, 6), and the random distractor dot is ignored: its state neither helps recover $a_t$ nor is predictable under $g_\phi$, so encoding it would inflate $\mathcal{L}_{\text{fwd}}$ without reducing $\mathcal{L}_{\text{inv}}$.
- **Control-dependent perception (sprite world):** post-hoc decoding from frozen embeddings shows the representation preserves controllable pose variables and averages out uncontrolled ones — with x/y-only control the decoded triangle is a blurred orientation-averaged blob at the right position; with full x/y/θ control it is sharp and oriented; with no actions the representation collapses to an average occupancy pattern.
- **Planning (Fig. 5):** SMWM ≈ SIGReg on TwoRoom, Reacher, Push-T; on OGBench-Cube — the only 3D contact-rich task, 5D action space — **SMWM 84% vs SIGReg 59%**. Both regularized variants dominate Forward-only and Random everywhere.
- **Extended goal offsets (Fig. 9):** with budget 2× offset, SMWM stays near-flat across horizons on TwoRoom and Cube where SIGReg degrades sharply (TwoRoom) or trails consistently (Cube); comparable on Reacher; Push-T degrades for both methods as offset grows.
- **Probing (Tab. 1):** both regularized methods recover physical state near-perfectly under MLP probes (e.g. TwoRoom agent position $R^2 = 1.000$ both). Linear probes: SMWM ahead on Push-T agent position (0.993 vs 0.954) and all three Cube quantities (cube position 0.998 vs 0.975; gripper opening 0.989 vs 0.961); SIGReg ahead on Reacher joint angles (0.999 vs 0.946) and Push-T block pose (orientation 0.931 vs 0.664). Forward-only is markedly poorer (e.g. Push-T agent position 0.293 linear) though rarely fully degenerate.
- **Latent geometry (Fig. 6 vs Fig. 10):** SMWM spectra show sharp elbows at the action dimension; TwoRoom PCs form a Cartesian map with an empty band where the wall is; Reacher's toroidal state space $T^2$ appears as a product-of-circles embedding in ~4 components (cylindrical 3D PC projections). SIGReg spectra show *no* elbow — expected, since it explicitly spreads variance toward isotropic Gaussianity — so its state information is probe-recoverable but not low-dimensional or directly interpretable.

## Ablations

- **Forward-only ($\lambda = 0$):** on dot world the encoder collapses all probe states to essentially one embedding (PC projection collapses to a point) — direct confirmation the inverse loss is the active anti-collapse mechanism. On benchmarks, forward-only planning is far below both regularized variants.
- **$\lambda$ sensitivity (Fig. 7):** $\lambda$ needs environment-level tuning — $\lambda = 0.1$ is optimal for TwoRoom but drives Reacher to near-random planning; final values chosen from a broad sweep (one model per $\lambda$; bands are binomial SE over 100 episodes, not seed variance).
- **Sprite-world action-repertoire ablation** (no / partial / full control) doubles as an ablation of what the inverse signal preserves — see Key results.

## Limitations

Paper's own: (1) assumes actions are recoverable from consecutive observations — fails when distinct actions cause the same visible change; (2) single-frame encoder cannot capture velocities or anything unidentifiable from one frame (multi-frame histories are a natural extension); (3) the action-aligned bias may discard information needed for downstream tasks that depend on aspects irrelevant to the *training* actions (authors call this "a feature rather than a bug"); (4) biased behavior policies may create action-correlated but uncontrollable distractors the encoder would capture; (5) usual offline world-model caveats — dataset coverage limits reachable plans, compounding rollout error, moderate-scale simulated tasks only. Honest-reader flags [analyst's view]: $\lambda$ requires per-environment tuning across two orders of magnitude (0.1–30), which undercuts the "one hyperparameter, simpler than SIGReg" pitch somewhat; the multi-step inverse identifiability theory they cite explicitly *does not* cover their continuous single-step setting (they acknowledge this carefully in App. H — discrete counterexamples exist where single-step inverse cannot distinguish states); and the SIGReg comparison uses history length 1 for both while LeWM originally used history 3 on Push-T/Cube, so the baseline may be slightly off its native configuration.

## Why it matters [analyst's view]

This is the cleanest entry yet on the vault's running "who solves JEPA collapse how" axis: frozen encoder ([[papers/baldassarre-2025-dino-world-models]], DINO-WM), distributional prior ([[papers/maes-2026-leworldmodel]], SIGReg), EMA/stop-gradient (V-JEPA 2, BYOL-line as in [[papers/guo-2022-byol-explore]]) — and now a *task-grounded* signal that regularizes by requiring action recoverability rather than prescribing latent geometry. The interesting claim is not just parity but *structure*: SIGReg provably prevents collapse but smears variance isotropically, whereas the inverse loss concentrates it on exactly the controllable subspace, and that structural difference appears to cash out precisely where planning is hardest (3D Cube, long horizons). The equivariance→homomorphism→translation chain is a rare case of a JEPA paper giving a mechanistic account of *why* its latent geometry looks the way it does, and the interventional linear-representation-hypothesis framing connects world-model latents to the LLM-geometry literature. The Reacher torus finding — periodic angles encoded as product-of-circles, visible as cylinders in PC space — directly echoes the circular population-code geometry found by probing in [[papers/joseph-2026-physics-video-world-models]] (shared co-author Balestriero), from the training side rather than the probing side. The philosophical framing (Umwelt, sensorimotor contingencies) is more than decoration: the sprite experiment operationalizes it — the same object is *represented differently depending on the agent's action repertoire* — which is a genuinely crisp empirical rendering of enactive perception.

## Open questions / things to verify

- Can $\lambda$ be auto-tuned (e.g. via loss balancing) to fix the per-environment sensitivity, or is the 0.1–30 spread intrinsic to action-space scale differences?
- Does single-step inverse break in environments matching the discrete counterexamples (Efroni et al., Lamb et al.) — and would the multi-step inverse extension the authors propose fix Push-T's long-horizon failure?
- How does SMWM scale beyond ~15M parameters and moderate simulated tasks — does inverse-dynamics regularization survive internet-scale video with sparse/noisy action labels?
- The claim that action-correlated distractors (from biased policies) would leak into the encoder is untested here — worth checking against the exogenous-noise benchmarks of the AC-State line.
- Concurrent work [37] proves a "no-unhealthy-collapse" theorem for auxiliary-head JEPAs in deterministic MDPs; does the guarantee formally cover this action-prediction instantiation in continuous spaces?

## Connections

- Direct baseline / architecture source: [[papers/maes-2026-leworldmodel]] — SMWM adopts LeWM's ViT-Tiny encoder + ViT-S predictor and its planning protocol wholesale, swapping SIGReg for $\lambda \mathcal{L}_{\text{inv}}$; head-to-head throughout.
- Contrasts with (frozen-encoder anti-collapse): [[papers/baldassarre-2025-dino-world-models]] — DINO-feature world models vs end-to-end training here.
- Contrasts with (EMA/self-predictive anti-collapse): [[papers/guo-2022-byol-explore]] — BYOL-style bootstrapped latent prediction; SMWM explicitly avoids EMA targets.
- Resonates with: [[papers/joseph-2026-physics-video-world-models]] — **author overlap: Randall Balestriero co-authored both**; circular/toroidal latent geometry of physical variables found by probing there, produced by training here.
- Related (Schölkopf world-model line): [[papers/lei-2024-spartan]] — *sibling note being written this session; link may be briefly dead until that note lands* — **author overlap: Bernhard Schölkopf co-authored both**.
- Adjacent (world models for manipulation): [[papers/higuera-2026-visuo-tactile-world-models]] — grounds latents in touch for contact-rich manipulation, complementary to grounding in actions.
- Naming-collision note: [[papers/ye-2026-world-action-models]] uses "WAM" for video-diffusion zero-shot policies (also framed as inverse dynamics on predicted futures); the WAM cited *in this paper* is Han & Yilmaz's Dreamer-based world-action model — different works, related inverse-dynamics spirit.
- Topic MOCs: [[topics/world-models]], [[topics/jepa]], [[topics/self-supervised-learning]], [[topics/representation-learning]]
- Author indices: _no vault index yet for Ivashkov, Balestriero, or Schölkopf_

## Selected quotes

> "This single regularizer addresses both issues: it prevents representation collapse and induces action-aligned representations. By forcing latent states to preserve information about the action underlying a transition, it biases the model toward the controllable degrees of freedom of the environment while discarding uncontrollable distractors." — Abstract

> "Unlike distributional regularizers that prescribe the geometry of the embedding space, inverse dynamics anchors the representation to a task-grounded quantity—the action." — §1

> "A simple way to make $a_t$ recoverable from $(z_t, z_{t+1})$ is to encode the action in the displacement vector $z_{t+1} - z_t \approx \rho(a_t)$. ... This mechanism does not enforce additivity, but it renders an additive, translation-like representation a low-complexity solution." — §4 / App. G

> "SMWM is the only entry that combines offline reward-free world-model training with inverse dynamics as the sole anti-collapse mechanism." — Tab. 5 caption, App. H
