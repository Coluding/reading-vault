---
type: paper
title: "SPARTAN: A Sparse Transformer World Model Attending to What Matters"
authors: ["Anson Lei", "Bernhard Schölkopf", "Ingmar Posner"]
year: 2024
venue: NeurIPS 2025
url: https://arxiv.org/abs/2411.06890
rw_id: 01ky5rdmn9prn0edk8bmpvzg6r
topics: [world-models, causal-discovery, interpretability]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# SPARTAN: A Sparse Transformer World Model Attending to What Matters

## TL;DR

SPARTAN (SPARse TrANsformer World model, Lei, Schölkopf & Posner — Oxford Applied AI Lab / MPI Tübingen) is a Transformer world model over object-factored tokens whose attention pattern is replaced by a **sampled hard (binary) mask with sparsity regularisation**, so that the learnt masks themselves become **state-dependent local causal graphs**: which objects causally influence which at *this* timestep. The core argument is that soft attention plus thresholding (the CoDA approach) is not enough — nothing forces attention mass onto only the true causal parents — whereas hard masking plus a penalty on the number of active edges makes the sparsest accurate model the loss-optimal one. The model further handles **interventions with unknown targets** via a learnable codebook of intervention tokens appended to the object tokens, and adapts to new environments at test time by gradient descent *in intervention-token space* rather than by finetuning weights. Across Interventional Pong, the CREATE physics environment, and real Waymo traffic data, SPARTAN matches or beats a SlotFormer-style Transformer on rollout prediction error while cutting the structural Hamming distance to ground-truth causal graphs from 6.29 to **1.17** (CREATE) and from 6.37 to **1.51** (Pong), and it is dramatically more robust to removing non-causal objects (error change 28.6% vs 138.2% on CREATE; 24.5% vs 1140.2% on Pong). The authors claim it is the first method to learn local causal structures beyond simple state-based settings.

## Context & motivation

Transformer world models over object-centric slots (SlotFormer being the state of the art the paper builds on) predict accurately but generalise and adapt poorly: prior work (CausalAgents, Roelofs et al.) showed motion-forecasting transformers are sensitive to removing scene entities that could not causally matter. The causal-ML framing offers a remedy: factorise dynamics into **independent causal mechanisms**, so that under the **Sparse Mechanism Shift hypothesis** (Schölkopf; Bengio et al.) a natural distribution change touches only a few mechanisms — a model mirroring the causal structure can then adapt by updating a small part while the rest stays invariant. Prior structured world models (VCD — the same authors' earlier work — and AdaRL) learn a *fixed global* causal graph, but for physical interactions the global graph is nearly fully connected and uninformative: any two billiard balls interact *at some point*. What is sparse is the **local** (state-dependent) graph — the two balls are causally linked only when about to collide. Existing local causal discovery methods (CAI, ELDEN, FCDL) work only on scalar state observations with fixed scene composition, and so cannot scale to image-derived object embeddings or scenes with varying object counts. SPARTAN's contribution is to instantiate sparsity-regularised local causal discovery *inside* a transformer, keeping the transformer's accuracy and its flexibility over variable numbers of tokens.

## Method

### Problem formulation

Training data are observation sequences from multiple environments $\{x_0, x_1, \dots, x_T, I\}$ where $x_t$ is the observation at timestep $t$ and $I$ an environment index; an "intervention" is a dynamics change (e.g. altering an object's friction), and its **targets are unknown** to the model. A representation model (assumed given) maps observations to a factored latent state $S = S^1 \times \dots \times S^N$, each $S^i$ an embedding of one entity (object slot or causal variable). The goal is a transition function $p(s^{t+1}\mid s^t)$ that captures the local causal structure. Formally, a causal graphical model factorises the joint as $p(v_1,\dots,v_N) = \prod_i p(v_i \mid Pa(V_i))$ ($Pa(V_i)$ = causal parents of $V_i$); an intervention $\mathcal{I}$ with targets $T^{\mathcal{I}}$ replaces the conditionals of targeted variables with intervened ones $p'$. The **local** causal graph $G^L$ for an observed state is the subgraph of the global graph obtained by deleting every edge $(i,j)$ that is "inactive" in the current neighbourhood ($V_j \perp\!\!\!\perp V_i \mid Pa_i(V_j)\setminus V_i$ locally) — so graph structure depends on the current state $s^t$. The discovery objective is the local analogue of sparsity-regularised causal discovery: minimise the *expected* number of edges and intervention targets over the data distribution, with graphs rebuilt at every timestep (FCDL gives identifiability conditions for this style of objective).

### Core idea

Replace soft attention with **Bernoulli-sampled hard attention masks** and penalise the expected number of active paths through the network. A soft-attention model can put non-zero attention anywhere as long as information from the true parent gets through, so thresholding attention can catch spurious edges or miss real ones; under hard masking plus an edge penalty, a model with spurious connections pays the sparsity cost and a model missing a true edge pays prediction cost — the training-loss optimum is exactly the model with only the correct causal edges.

### Architecture / algorithm

**Single layer.** From object tokens $s^t_{1:N}$, compute keys/queries/values $\{k_i, q_i, v_i\}$ by linear projection as usual. Then *sample* a binary adjacency matrix

$$A_{ij} \sim \mathrm{Bern}\big(\sigma(q_i^\top k_j)\big),$$

where $\mathrm{Bern}$ is the Bernoulli distribution and $\sigma$ the sigmoid — i.e. the query-key logit parameterises the *probability that edge $j \to i$ exists*, making the graph state-dependent through the key-query pairs. Hidden features use standard scaled dot-product attention with $A$ as a multiplicative mask:

$$h_i = \sum_j \frac{A_{ij}\,\exp(q_i^\top k_j/\sqrt{d_k})\, v_j}{\sum_j \exp(q_i^\top k_j/\sqrt{d_k})}, \qquad \hat{s}^{t+1}_i = \mathrm{MLP}(h_i + s_i),$$

with $d_k$ the key dimension and $\hat{s}^{t+1}_i$ the next-step prediction for object $i$. When $A_{ij}=0$, no information flows from token $j$ into the prediction for token $i$ — the mask *disallows* flow from non-parents rather than merely down-weighting it, so $A_{ij}=1$ is interpretable as $s^t_j \in Pa_L(s^{t+1}_i)$. Sampling is made differentiable with the Gumbel-softmax trick.

**Multi-layer: the path matrix.** Stacking $L$ layers breaks the single-layer interpretation: token $i$ can reach token $j$ via a multi-hop route $i \to k \to j$ across layers with no single edge $(i,j)$ present. SPARTAN therefore counts **paths**:

$$\bar{A} = (A^{L+1} + \mathbb{I})\cdots(A^2 + \mathbb{I})(A^1 + \mathbb{I}),$$

where $A^{\ell}$ is layer $\ell$'s sampled adjacency and the identity $\mathbb{I}$ accounts for residual connections (a token always reaches its own next-layer copy). $\bar{A}_{ij}$ is then the number of directed paths from token $j$ to token $i$ through the whole stack, and $s^t_j$ is a local causal parent of $s^{t+1}_i$ iff $\bar{A}_{ij} \geq 1$. This is what gets regularised — penalising per-layer edges alone would not prevent multi-hop leakage.

**Interventions with unknown targets.** For multi-environment training the model keeps a learnable codebook of **intervention tokens** $T_{1:K} \in \mathbb{R}^{d \times K}$ ($d$ = state-token dimension, $K$ = number of training interventions). The token $T_I$ for the current environment is appended as an extra $(N{+}1)$-th input token and everything proceeds as above; $\bar{A}_{i(N+1)} = 0$ then means no path from the intervention token to $s^{t+1}_i$, i.e. object $i$ is **not** an intervention target in this environment — the model must *discover* the targets, and the sparsity penalty (which covers these edges too) prunes spurious ones. This is an extension, not a requirement: single-environment SPARTAN works without it.

### Derivations / training objective

The training objective is sparsity-regularised expected loss:

$$\min_\theta \; \mathbb{E}_{s^t, s^{t+1}, I}\Big[\mathcal{L}(\hat{s}^{t+1}) + \lambda_s\,|\bar{A}|\Big],$$

where $\theta$ covers Transformer parameters and intervention tokens, $\mathcal{L}$ is a prediction loss such as MSE, $\lambda_s$ the regularisation weight, and $|\bar{A}|$ — because the intervention token is an input — counts causal edges *and* intervention targets together.

**Lagrangian relaxation (the part that makes it trainable).** Training is sensitive to $\lambda$: too high causes mode collapse. Since deleting truly non-causal edges should not hurt prediction, the paper reformulates as constrained optimisation — minimise the expected number of paths subject to prediction error staying below a target $\tau$:

$$\min_\theta \; \mathbb{E}\big[|\bar{A}|\big] \quad \text{s.t.} \quad \mathbb{E}\big[\mathrm{MSE}(s^{t+1}, \hat{s}^{t+1})\big] \leq \tau,$$

with $\tau$ set to the loss achieved by a fully connected model. Lagrangian relaxation gives the min-max objective

$$\max_{\lambda > 0}\;\min_\theta\; \mathbb{E}\big[|\bar{A}|\big] + \lambda\big(\mathbb{E}[\mathrm{MSE}(s^{t+1}, \hat{s}^{t+1})] - \tau\big),$$

solved by alternating gradient steps on $\theta$ and $\lambda$. Positivity of $\lambda$ is kept via multiplicative updates $\lambda \leftarrow \alpha \cdot \exp(\mathrm{MSE} - \tau) \cdot \lambda$ ($\alpha$ = step size): $\lambda$ grows whenever error exceeds the target, re-weighting accuracy over sparsity. A moving-average estimate of $\mathrm{MSE} - \tau$ stabilises this, and the loss is rewritten as $(\mathrm{MSE} - \tau) + |\bar{A}|/\lambda$ for numerical stability. The authors liken it to the GECO scheme for tuning VAE KL weights. The emergent schedule: early in training $\lambda$ stays low and edge count *rises* while dynamics are learnt; once error drops below $\tau$ the sparsity pressure automatically ramps and redundant edges are pruned without losing accuracy (App. A, Fig. 5).

### Training procedure

Simulated domains: object slots obtained by encoding each ground-truth-masked object with a VAE (32-dim tokens for Pong, 64 for CREATE; velocity is exposed by rendering 2-step position trails into the object image). SPARTAN/Transformer: embedding dim 512, 3 transformer layers, MLP hidden 512 (Pong) / 1024 (CREATE), 3 MLP layers per transformer layer, Adam, lr 5e-5. Single GPU (V100 or RTX 6000), convergence within 3 days. Traffic: MTR (Motion Transformer) is the base architecture — self-attention over tokenised map polylines and vehicle trajectories, then multi-query cross-attention producing a Gaussian-mixture over motion modes — with **all attention layers swapped for the sparsified version**; ~1000 tokens per scene, 4 GPUs, under two weeks (baseline MTR under one week).

### Inference / adaptation

Rollouts are produced by iterating the one-step model (example rollouts in Fig. 2 visualise the sampled graphs per step). **Test-time adaptation to an unknown intervention**: given a few observation sequences *without* an environment index, freeze the model and optimise a free intervention token $T_{\text{adapt}} \in \mathbb{R}^d$ by gradient descent in token space to fit the observed data. Because $T_{\text{adapt}}$ need not lie in the discrete training codebook, this generalises to unseen environments, including compositions of seen interventions.

## Experimental setup

- **Datasets**: *Interventional Pong* (CITRIS benchmark, modified so interventions are dynamics changes — curved ball trajectories, paddle inversion, mid-field friction, gravity, altered bounce; 4 objects: two paddles, ball, score; 32×32×3 images; envs 0–7 seen, 8–10 unseen compositions of seen interventions). *CREATE* (2D physics: red/blue/green balls, wall, ladder, cannon; 84×84×3; interventions include disabled gravity, reversed ladders, elastic walls, disabled cannon; unseen envs change scene composition — different object counts). *Traffic* (Waymo Open Dataset, real; ~800 map polylines + ~50 vehicles per scene, 1s history → 9s ego-vehicle motion prediction; no interventions since data is not simulated).
- **Baselines**: *Transformer* (SlotFormer instantiation with one-step context, ground-truth-mask slots; MTR for Traffic), *Global Graph* (VCD/AdaRL-style masked-MLP ensemble with a learnable fixed adjacency), *CoDA*-style attention thresholding, *Local Attn* (MTR's spatial K-NN attention mask, Traffic only), *ACD* (amortised causal discovery, GNN backbone, App. C).
- **Metrics**: rollout prediction error; **Structural Hamming Distance (SHD)** between learnt and ground-truth causal graphs (human-labelled graphs from CausalAgents for Traffic); percentage change of prediction error when non-causal entities are removed; few-shot (5-trajectory) adaptation error.

## Key results

- **Graph accuracy (Table 1, SHD, lower better)**: SPARTAN 1.51 / 1.17 / 6.84 (Pong / CREATE / Traffic) vs Transformer 6.37 / 6.29 / 10.6 and Global Graph 5.42 / 8.77 / n.a. — and the Transformer baseline's threshold was chosen *using ground truth* to give its best SHD, which is not available in practice.
- **Prediction (Table 1)**: SPARTAN 8.60 / 246.6 / 0.52 vs Transformer 8.83 / 265.5 / 0.51 — sparsity costs essentially nothing in accuracy and often helps.
- **Robustness (Table 2, % error change when non-causal entities removed)**: SPARTAN 24.5±4.4 (Pong), 28.6±1.7 (CREATE), 13.9±0.2 (Traffic) vs Transformer 1140.2±15, 138.2±5.6, 32.8±0.4. Global Graph is competitive on Pong (22.2±1.3, where the true graph is nearly static) but degrades on CREATE (73.6±1.3); MTR's K-NN Local Attn (25.5±0.3) still trails SPARTAN on Traffic. The paper notes CausalAgents reports 25–38% for a wide range of architectures and 22% even with ground-truth-informed data augmentation — SPARTAN's 13.9% beats that without ground truth.
- **Few-shot adaptation (Fig. 4)**: with 5 trajectories from an unknown environment, SPARTAN consistently achieves the lowest errors, approaching the lower bound of a model given the true environment index, including on unseen environments (composed interventions in Pong; changed object counts in CREATE, where Global Graph cannot even be applied).
- **Qualitative**: learnt graphs match ground truth (ball → paddles as paddles track the ball; paddle → ball at collision; ball → score at boundary crossings); Transformer produces spurious edges like "Score → Ball". On Traffic, SPARTAN attends to vehicles in neighbouring lanes consistent with human labels — spatial-proximity heuristics fail here because relevant vehicles can be far ahead while nearby oncoming ones are irrelevant.
- **vs ACD (App. C, Table 7, SHD)**: SPARTAN 1.50 / 1.17 vs ACD 10.27 / 39.77 and even Sparse ACD with oracle sparsity prior 2.67 / 5.45.

## Ablations

No dedicated ablation section; the load-bearing comparisons play that role: (1) **hard attention + sparsity vs soft attention + thresholding** (the CoDA comparison) shows regularised hard attention is *necessary* for accurate local causal discovery in complex, observation-based environments; (2) **local vs global graphs** (Global Graph baseline) shows fixed graphs suffice only when the true structure is nearly time-invariant (Pong) and fail as state-dependence grows (CREATE); (3) **learned vs heuristic masks** (Local Attn K-NN) shows spatial heuristics are strictly worse; (4) **App. D**: replacing ground-truth-mask slots with learned Slot Attention representations on Pong preserves the finding — SPARTAN pred. error 7.21±0.69 / SHD 2.69±0.02 vs Transformer 6.90±0.83 / SHD 8.52±0.02.

## Limitations

**Paper's own**: (1) no theoretical identifiability statement for local causal graphs when scene composition varies across samples (the sparsity-regularisation grounding leans on FCDL's guarantees for a narrower setting); (2) adaptation is confined to the learnt intervention space — genuinely novel test-time behaviours (e.g. a teleporting ball) may have no corresponding token, where weight finetuning would eventually win given enough data; they suggest procedurally generated interventions (domain-randomisation-style) as a fix; (3) reliance on pre-disentangled object representations — evaluations use ground-truth segmentation masks (justified as needed for unambiguous SHD evaluation), though App. D shows pretrained slots also work.

**[analyst's view]** Additional flags: the graph-evaluation environments are 2D and small (4–9 objects); Traffic is the only real-world domain and there the SHD gap (6.84 vs 10.6) is narrower than in simulation. Bernoulli sampling + Gumbel-softmax over every token pair per layer, plus the Lagrangian schedule, adds training cost and fragility (SPARTAN takes up to 2× the baseline's training time on Traffic), and the method's sensitivity to $\lambda$ is managed rather than eliminated. Interventions in training data are required for the intervention-token machinery, which is a strong data assumption outside simulators.

## Why it matters [analyst's view]

This is the cleanest existing demonstration that a transformer's attention can be turned into an *explicit, evaluable causal object* rather than a post-hoc interpretability artifact — the mask is the model's actual information-flow constraint, so "reading off the graph" is exact, not heuristic. For the vault's world-models thread it is the structural-interface counterpoint to pixel/latent-space world models: [[papers/chen-2026-actionable-simulators]] marshals SPARTAN as its exemplar of exposing causal structure for reasoning instead of merely rendering futures, and its robustness numbers give empirical teeth to that position. The intervention-token adaptation scheme is also a notably lightweight alternative to weight finetuning — conceptually adjacent to (but structurally different from) [[papers/gao-2025-adaworld]]'s latent-action interface for few-shot world-model adaptation: AdaWorld adapts *action conditioning* at scale in pixel space, SPARTAN adapts *mechanism shifts* in a factored latent space with unknown targets. The Sparse Mechanism Shift framing is the connective tissue: if real distribution shifts are sparse in mechanism space, models that localise mechanisms adapt for free. The open question is whether any of this survives the move from ground-truth-masked 2D slots to messy learned representations at scale.

## Open questions / things to verify

- Does the path-matrix regularisation stay stable at many layers / thousands of tokens, where $\bar{A}$ entries can grow combinatorially?
- How brittle is the Lagrangian $\tau$ (set from a fully connected model's loss) when that reference model itself overfits?
- The authors' proposed follow-up: can local sparsity *induce* disentangled causal representations if the encoder is trained jointly with the dynamics? Worth watching for a sequel.
- FCDL (Hwang et al., ICML 2024) provides the identifiability theory this leans on — worth reading to know what actually transfers to the varying-composition setting. `_needs note_`
- SlotFormer and VCD (the authors' own prior fixed-graph model) are the direct ancestors; neither is in the vault. `_needs note_`

## Connections

- Builds on: SlotFormer (transformer dynamics over object slots), VCD / AdaRL (fixed global causal graphs via sparsity regularisation) — `_needs note_`, not in vault
- Cited as exemplar in: [[papers/chen-2026-actionable-simulators]] (structured/causal interfaces for world models)
- Contrasts with: [[papers/gao-2025-adaworld]] (few-shot world-model adaptation via latent actions rather than intervention tokens)
- Survey context: [[papers/ding-2024-world-models-survey]] (world-model taxonomy this sits inside)
- Topic MOCs: [[topics/world-models]], [[topics/interpretability]]
- Author indices: none yet in vault (Anson Lei, Bernhard Schölkopf, Ingmar Posner — create on next paper by these authors)

## Selected quotes

> "we postulate that sparsity is a critical ingredient for the discovery of such local structures" — Abstract

> "applying a threshold on the attention values may 'catch' spurious edges or miss necessary edges. Instead, we need a model that masks the information flow between the tokens and penalises connections between tokens." — §3

> "The path matrix A¯ has the property that A¯ij is the number of paths that lead from j to i. In this case, s t j is a local causal parent of s t+1 i iff A¯ij >= 1." — §3.1

> "To the best of our knowledge, SPARTAN is the first method that learns local causal structures beyond simple state-based settings." — §1
