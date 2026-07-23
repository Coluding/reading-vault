---
type: paper
title: "GenRL: Multimodal-foundation world models for generalization in embodied agents"
authors: ["Pietro Mazzaglia", "Tim Verbelen", "Bart Dhoedt", "Aaron Courville", "Sai Rajeswar"]
year: 2024
venue: NeurIPS 2024
url: https://arxiv.org/abs/2406.18043
rw_id: 01ky5twq8zrwfg04rw9m49fsrw
topics: [world-models, reinforcement-learning, representation-learning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# GenRL: Multimodal-foundation world models for generalization in embodied agents

> Source note: full text fetched via Readwise (PDF parse). The author block of the PDF was rendered as an image; the full author list was confirmed from the arXiv abstract page (`citation_author` metadata, fetched 2026-07-23). Figures and the appendix plots (Figs. 4–13) are described but their bar values were not extractable — per-task generalization numbers (Appendix K) are a known gap.

## TL;DR

GenRL (NeurIPS 2024, Mazzaglia et al., Ghent/Mila/ServiceNow) learns *multimodal-foundation world models* (MFWMs): the joint video–language embedding space of a frozen foundation VLM (InternVideo2) is connected and aligned to the discrete latent space of a generative world model trained on **vision-only** embodied data — no language annotations, no rewards. A task given as a language or video prompt is embedded by the VLM, translated by a learned *connector* into a target sequence of world-model latent states, and the policy is trained purely in imagination to match that latent trajectory, with reward = cosine similarity between projected imagined and target latents. On 35 reward-free tasks across four locomotion domains and the Kitchen manipulation domain, GenRL scores 0.80 ± 0.02 overall (min-max scaled, 1.0 = expert) on in-distribution language-to-action, vs 0.70 for the best model-based baseline (WM-CLIP) and ≤0.41 for model-free VLM-reward baselines. A *data-free* variant learns new task policies after pretraining with zero access to any dataset — sampling initial states inside the model — at only a slight performance cost, which the authors frame as groundwork for foundation-model-style pretrain-once/adapt-freely RL.

## Context & motivation

Scaling RL to many tasks is bottlenecked by per-task reward design. Language is a natural task interface, and prior work uses VLM similarity scores (e.g. CLIP score between observations and a text prompt) as rewards — but these approaches generally need VLM fine-tuning, prompt hacking, or environment modifications, because of the domain gap between web imagery and embodied observations. Worse, embodied domains typically lack multimodal data to fine-tune on: there is no natural language description of motor currents or joint torques, and labelling agent interactions is expensive. The paper's question: *how do you leverage foundation VLMs for generalization in embodied domains without any language annotations?*

GenRL's answer combines two mature lines: generative world models for RL (Ha & Schmidhuber; Dreamer-line work — the paper builds on DreamerV3-style discrete latent models and learning-in-imagination) and contrastive video-language models (InternVideo2). The stated contributions: (i) MFWMs, which connect/align a foundation VLM representation with a world-model latent space using vision-only data; (ii) an RL objective that grounds vision/language prompts as latent-trajectory targets and learns behaviors in imagination; (iii) *data-free policy learning* — task adaptation with no data access post-pretraining. The paper distinguishes itself from DynaLang (which shares one representation between vision and language inside the world model, needing language data) by keeping the world model vision-only and bolting the multimodal space on via connector/aligner.

## Method

### Problem formulation

The agent receives image observations $x \in \mathcal{X}$ and acts with $a \in \mathcal{A}$. A task $\tau$ is specified either visually ($x^\tau$, an image/video prompt) or in language ($y^\tau \in \mathcal{Y}$). **No reward signal is assumed for training**; ground-truth rewards exist only for evaluation. The training corpus is a large, reward-free, annotation-free set of visual trajectories (mixed expert replay + exploration data).

### Core idea

Train a vision-only latent world model; separately train a *connector* that maps frozen-VLM embeddings to world-model latent state sequences and an *aligner* that closes the VLM's vision–language modality gap using noise around visual embeddings — so at test time a language prompt becomes a latent target trajectory, and RL in imagination maximizes similarity to it.

### Architecture / algorithm

**1) World model.** A task-agnostic model of latent dynamics in a compact discrete latent space $\mathcal{S}$; latent states $s$ are sampled from independent categorical distributions (straight-through gradient estimation through the sampling). Components:

- Encoder: $q_\phi(s_t \mid x_t)$
- Sequence model: $h_t = f_\phi(s_{t-1}, a_{t-1}, h_{t-1})$ (a linear GRU cell)
- Decoder: $p_\phi(x_t \mid s_t)$
- Dynamics predictor: $p_\phi(s_t \mid h_t)$

trained with:

$$\mathcal{L}_{\phi} = \sum_{t} \underbrace{D_{\text{KL}}\big[q_{\phi}(s_t|x_t)\,\|\,p_{\phi}(s_t|s_{t-1}, a_{t-1})\big]}_{\text{dyn loss}} \; - \; \underbrace{\mathbb{E}_{q_{\phi}(s_t|x_t)}\big[\log p_{\phi}(x_t|s_t)\big]}_{\text{recon loss}}$$

where $p_\phi(s_t|s_{t-1},a_{t-1})$ abbreviates $p_\phi(s_t | f_\phi(s_{t-1},a_{t-1},h_{t-1}))$. The KL term forces the action-conditional dynamics prediction to match what the encoder infers from the actual next frame (so imagination stays truthful); the reconstruction term forces latents to carry enough information to redraw the frame. **Key deviation from the Dreamer RSSM**: encoder and decoder are *not* conditioned on the recurrent state $h_t$. Each latent $s_t$ therefore encodes exactly one observation (temporal information lives only in $h_t$), making the encoder a "probabilistic visual tokenizer" — this matters because the connector (below) must be able to hit these latents from a VLM embedding without access to the world model's history.

**2) Connecting the VLM.** The frozen VLM provides a vision embedder $e^{(v)} = f^{(v)}_{PT}(x_{t:t+k})$ over $k$-frame clips and a language embedder $e^{(l)} = f^{(l)}_{PT}(y)$; GenRL uses InternVideo2 with $k=8$ (image-language models are the $k=1$ special case). Two new modules:

$$\text{Connector: } p_{\psi}(s_{t:t+k}\mid e), \qquad \mathcal{L}_{\text{conn}} = \sum_{t} D_{\text{KL}}\big[\, p_{\psi}(s_t|s_{t-1}, e)\,\|\,\text{sg}(q_{\phi}(s_t|x_t))\,\big]$$

$$\text{Aligner: } e^{(v)} = f_{\psi}(e^{(l)}), \qquad \mathcal{L}_{\text{align}} = \| e^{(v)} - f_{\psi}(e^{(l)}) \|_2^2$$

with $\text{sg}(\cdot)$ = stop-gradient (the world model is not perturbed by the connector's training). The connector is a GRU-based sequence model — deliberately symmetric with the world model's own dynamics module rather than a transformer — that predicts the world-model latent sequence for a clip *from the clip's VLM embedding*. It is trained on visual embeddings only.

**Why an aligner exists.** Contrastive VLMs exhibit a *multimodality gap*: vision and language embeddings of the same content occupy misaligned regions of the sphere. The connector trained on $e^{(v)}$ can only generalize to the corresponding $e^{(l)}$ if the angle $\theta$ between them satisfies $\cos\theta > c$ for some constant $c$. Prior fixes inject noise into visual embeddings *during connector training*, effectively growing $c$ (a blurrier connector). GenRL instead trains a separate aligner network that maps points *sampled around* $e^{(v)}$ (via noise — hence trainable from vision-only data) back close to $e^{(v)}$. At test time the language embedding, being near its visual counterpart, gets pulled onto the visual manifold the connector was trained on. Advantages the paper claims: the connector stays noise-free (higher prediction accuracy on visual inputs), and the aligner is a cheap, swappable module (re-train for different noise levels without touching the connector). The aligner is a small U-Net whose bottleneck is half the embedding size; embeddings are 768-dimensional.

**3) Task specification and learning in imagination.** Given prompt embedding $e_{\text{task}} = f_{PT}(\cdot)$ (language embeddings pass through the aligner first), the connector generates target latent states $s_{t:t+k}$ — which can even be decoded to pixels to *visualize* how the model interpreted the prompt before any training. The policy objective is trajectory matching, posed as divergence minimization between policy-visited state distribution and the prompt-generated target trajectory:

$$\theta = \arg\min_{\theta}\; \mathbb{E}_{a_t \sim \pi_{\theta}(s_t)} \Big[ \sum_t \gamma^t\, \text{distance}\big(\, p_{\phi}(s_{t+1}|s_t, a_t)\, \|\, p_{\psi}(s_{t+1}|e_{\text{task}})\,\big) \Big]$$

KL is the natural distance here, but in practice cosine distance between *linear projections* of latents learns much faster and more stably, giving the reward:

$$r_{\text{GenRL}} = \cos\big(g_{\phi}(s_{t+1}^{\text{dyn}}),\; g_{\phi}(s_{t+1}^{\text{task}})\big), \quad s_{t+1}^{\text{dyn}} \sim p_{\phi}(s_{t+1}|s_t, a_t),\; s_{t+1}^{\text{task}} \sim p_{\psi}(s_{t+1}|e_{\text{task}})$$

where $g_\phi$ is the **first linear layer of the world model's decoder** — a projection that already exists and organizes latents by visual content, so no extra head is trained. For stability the target $s^{\text{task}}$ uses the mode of the connector's distribution. An actor-critic $\pi_\theta(a_t|s_t)$, $v_\theta$ maximizes this reward in imagination, following DreamerV3 (v1) practices: two-hot categorical critic over a $\lambda$-return target $R^\lambda_t$ (reward plus discounted bootstrapped value mixture), and return scaling in the actor loss.

**4) Temporal alignment ("best matching trajectory").** Trajectory matching implicitly assumes the policy starts where the target starts — false in general (an agent lying down and told to run needs more get-up steps than the VLM's 8-frame window). Fix, inspired by best-path decoding in CTC speech recognition: (1) slide the first $b$ target states along the imagined trajectory to find the timestep $t_a$ of best alignment (highest reward); (2) states before $t_a$ are compared to the target's initial state, states $k$ steps after $t_a$ to target state $s_{t+k}$. They fix $b=8$ (the VLM frame span), a compromise between $b=1$ (initial state only) and $b=$ horizon (no alignment); ablated in Appendix E.

### Derivations / why it works

The load-bearing arguments are the geometric alignment argument (connector generalizes from $e^{(v)}$ to $e^{(l)}$ iff $\cos\theta > c$; the aligner enlarges the effective basin without degrading the connector — Fig. 3) and the reduction of trajectory matching (Eq. 2, from model-based imitation-by-trajectory-matching literature) to a per-step cosine reward (Eq. 3). No formal proofs; the aligner ablation (below) is the empirical justification.

### Training procedure

- Datasets: per-domain corpora of ~1.8M–3.6M observations (walker 2.5M, cheetah 1.8M, quadruped 2.5M, kitchen 3.6M, stickman 2.5M, minecraft 4M), each a mix of task-specific replay buffers from DreamerV3 agents and unstructured exploration data from Plan2Explore. **No rewards, no text.**
- MFWM: batch 48, sequence length 48, GRU 1024 units, dense hidden 1024, 4 MLP layers, CNN multiplier 48; 500k gradient steps ≈ 5 days on a 16GB V100. VLM embeddings precomputed and stored with the datasets.
- Actor-critic: batch 32, sequence length 32; 50k gradient steps (<5h; <3h data-free; convergence typically by 10k). Other hyperparameters follow DreamerV3 v1 (Jan 2023).
- Prompts were tuned for InternVideo2 (e.g. "spider running fast" for quadruped — VLMs are biased toward human actions, so naming the embodiment and adding "fast"/"clean" helps); the same prompts mostly also helped SigLIP baselines.

### Inference / sampling

Offline mode: real observation sequences initialize latent states, then imagination rollouts + $r_{\text{GenRL}}$. **Data-free mode**: initial latents are sampled *inside* the model — uniform samples from the discrete latent space mixed with states generated by randomly sampling the connector (random embeddings mapped through aligner → connector), then a 5-step warmup rollout with mixed policy/random actions so the GRU acquires velocity information. No dataset access at all after pretraining.

## Experimental setup

- **Domains/tasks**: 35 reward-free tasks across Walker, Cheetah, Quadruped, and the newly introduced **Stickman** (a Walker variant with 2-joint arms + head, 10 joints total — upper-body tasks like boxing/handstand without full humanoid data cost), plus Kitchen (microwave, light, burner, slide).
- **Baselines**: model-free offline RL (IQL, TD3+BC, TD3) with VLM-similarity rewards in two flavors — image-language (SigLIP-B) and video-language (InternVideo2, same VLM as GenRL); plus **WM-CLIP**, a model-based "reversed connector" baseline that predicts VLM embeddings *from* world-model states ($\hat{e}^{(v)}_t = f_\psi(s_t, h_t)$, 4-layer MLP, hidden 1024) and rewards imagined states by cosine similarity to the prompt embedding. Same world model as GenRL for fairness.
- **Protocol**: 500k gradient steps for baselines (GenRL's actor-critic needs only 50k), 20 evaluation episodes, 10 seeds; scores min-max rescaled (0 = random policy, 1 = expert).

## Key results

- **Language-to-action, in-distribution (Table 1)**: GenRL 0.80 ± 0.02 overall vs WM-CLIP-V 0.70 ± 0.04, TD3-V 0.41 ± 0.05, best image-language baseline (TD3+BC-I) 0.33 ± 0.04. Largest wins on dynamic tasks (cheetah run 0.74 vs 0.56 WM-CLIP-V; quadruped run 0.86 vs 0.61; walker walk 1.01). On some static Kitchen tasks other methods occasionally win (kitchen burner: WM-CLIP-V 0.78 vs GenRL 0.62) — GenRL's inferred targets are "slightly in motion" even for static prompts.
- **Language-to-action generalization** (tasks absent from training data, Fig. 4): same ordering; GenRL significantly beats all model-free approaches, near specialist performance in quadruped/cheetah. Conservative offline RL (IQL, TD3+BC) degrades most — imitating dataset segments doesn't reach unseen tasks.
- **Video-to-action (Fig. 5)**: video prompts perform on par with language prompts on the same tasks; the VLM generalizes across visual styles (drawings, AI-generated), viewpoints, and morphologies.
- **Data-free policy learning (Fig. 6)**: slight overall drop vs offline GenRL but still above all baselines; per-domain differences minimal, Kitchen actually improves. Removes CPU–GPU data transfer, roughly halving policy-training time (convergence often within ~30 min).
- **LIV comparison (Kitchen, Table 6)**: GenRL 0.76 ± 0.08 vs IQL+LIV 0.20, WM-CLIP+LIV 0.17 — non-fine-tuned LIV rewards fail, consistent with LIV's own paper.
- Stated summary: video-language > image-language for dynamic tasks; model-based > model-free; connector-aligner direction > reversed connector.

## Ablations

- **Aligner (Table 5)**: removing it collapses GenRL from 0.76 ± 0.01 to 0.17 ± 0.00 overall — unaligned language embeddings are too far from the visual embeddings the connector was trained on. Conversely, adding the aligner to baselines' cosine-similarity rewards (TD3-V, WM-CLIP-V + aligner) changes little (0.40→0.42, 0.67→0.68) — the aligner specifically rescues the *connector input*, not similarity scoring; it neither improves nor hurts the raw VLM signal.
- **Temporal alignment (Appendix E, Fig. 8)**: best-matching-trajectory ablated over $b$; $b=8$ chosen (plot values not extractable from the parse).
- **Training data distribution (Sec. 4.3, Fig. 7, walker)**: full dataset best; varied exploration data (Plan2Explore) is the most important subset; task-data usefulness tracks complexity — "run data" transfers more broadly than "walk"/"stand" data; near-static "stand data" only supports simple poses.

## Limitations

Paper-acknowledged: inherits the multimodality gap and prompt-tuning sensitivity from VLMs (mitigated by aligner + decodable targets); inherits reconstruction dependence from the world model — in Minecraft the MFWM produces blurry reconstructions even when scaled up, making precise prompts ("attack a zombie") ungroundable while coarse ones ("navigate to a beach") work; static prompts yield slightly-moving targets (fixable with length-1 targets, deliberately not done for simplicity); behaviors beyond the VLM's 8-frame temporal comprehension are hard; complex behaviors require expert data in pretraining (data-driven ceiling); Kitchen out-of-distribution manipulation fails (an unopened cabinet in the dataset can't be opened offline). Honest-reader additions [analyst's view]: all environments are simulated with 64×64 observations and mostly dm_control-scale dynamics — no real-robot results; evaluation prompts were tuned to the very VLM used for rewards, so prompt sensitivity is partially hidden; the min-max scaling makes cross-paper comparison hard.

## Why it matters [analyst's view]

GenRL is the cleanest existing demonstration that a frozen web-scale VLM and a domain-specific world model can be stitched together *post-hoc* with only vision data — the connector/aligner recipe sidesteps both VLM fine-tuning and multimodal data collection, the two standard blockers. The data-free result is conceptually the most important: it decouples task adaptation from dataset access, which is the RL analogue of prompt-based adaptation in LLMs, and the paper explicitly frames it as groundwork for "foundation models in embodied RL". In the vault this sits squarely in the imagination-based learning thread: [[papers/chen-2026-actionable-simulators]] cites GenRL twice as an exemplar — of *self-evolution* (refining policies and representations through imagined experience) and of imagination-centric generalization in interaction-starved regimes — essentially adopting GenRL as evidence for its "actionable simulator" thesis. The aligner ablation (0.76 → 0.17) is a crisp, reusable finding about the multimodality gap: alignment matters enormously when embeddings are *inputs to another network*, and not at all for similarity scoring. The dependence on varied exploration data (Sec. 4.3) also links this to the exploration literature the vault tracks (e.g. [[papers/guo-2022-byol-explore]] as an alternative intrinsic-exploration source to the Plan2Explore data GenRL consumes). The reconstruction bottleneck they hit in Minecraft is exactly the gap diffusion-based world models ([[papers/ding-2024-diffusion-world-model]], [[papers/jiang-2025-world4rl]]) attack from the other side.

## Open questions / things to verify

- Per-task generalization and data-free numbers (Appendix K) and the temporal-alignment ablation values (Fig. 8) were in figures the PDF parse dropped — re-check against the paper/website (mazpie.github.io/genrl) if needed.
- How far does the connector/aligner recipe carry with stronger backbones (transformer/diffusion connector, larger VLMs)? The authors themselves flag GRU + reconstruction as the weak link.
- Composability: the proposed future work of an LLM composing GenRL behaviors for long-horizon tasks is untested here.
- Does the aligner trick transfer to other modality-gapped settings (e.g. audio-language, action-language)?

## Connections

- Builds on: DreamerV3-style discrete latent world models and learning-in-imagination (Hafner et al.); InternVideo2 video-language embeddings; Plan2Explore exploration data.
- Cited as exemplar by: [[papers/chen-2026-actionable-simulators]] (self-evolution via imagined experience; imagination-based generalization)
- Contrasts with: [[papers/ding-2024-diffusion-world-model]], [[papers/jiang-2025-world4rl]] — diffusion world models addressing the reconstruction/rollout-fidelity weaknesses GenRL hits in Minecraft
- Related: [[papers/guo-2022-byol-explore]] — intrinsic exploration in latent world models; GenRL's generalization empirically depends on exactly this kind of varied exploration data
- Topic MOCs: [[topics/world-models]], [[topics/reinforcement-learning]], [[topics/representation-learning]]
- Author indices: [[authors/pietro-mazzaglia]]

## Selected quotes

> "GenRL, allows one to specify tasks through vision and/or language prompts, ground them in the embodied domain's dynamics, and learn the corresponding behaviors in imagination." — Abstract

> "we define it as the ability to generalize to new tasks, after pre-training, by learning a policy completely in imagination, with no access to data (not even to the pre-training dataset)." — §4.2, on data-free policy learning

> "the aligner mechanism is crucial in GenRL's functioning." — Appendix F

> "The results confirm that a diverse data distribution is crucial for task success, with the best performance achieved by using the complete dataset, followed by the varied exploration data." — §4.3
