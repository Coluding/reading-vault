---
type: paper
title: "Clockwork Variational Autoencoders"
authors: ["Vaibhav Saxena", "Jimmy Ba", "Danijar Hafner"]
year: 2021
venue: arXiv preprint (arXiv:2102.09532)
url: https://arxiv.org/abs/2102.09532
rw_id: 01ky7gga6h39tfqgq0393fhf5r
topics: [world-models, video-generation, generative-models, recurrence]
priority: medium
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# Clockwork Variational Autoencoders

> Venue note: the fetched text does not state a publication venue; frontmatter uses "arXiv preprint". Figures and appendix figures E.1/F.1/G.1/H.1 were referenced but not visible in the fetched text; all numbers below come from the running text and tables that were present.

## TL;DR

The Clockwork VAE (CW-VAE) is a hierarchical latent-dynamics video prediction model in which each level of the latent hierarchy ticks at an exponentially slower fixed clock speed than the level below (factor $k$ per level). Because the KL cost of writing information into a level is paid only at its active timesteps, slowly changing content (e.g. maze wall textures, digit identities) migrates up the hierarchy for "free", while fast content (camera position, digit motion) stays in the fast lower levels — and the paper verifies this separation experimentally. Predictions are rolled out purely in latent space (no generated frames fed back in), which avoids the error accumulation of image-autoregressive models. On four datasets (MineRL Navigate, KTH Action, GQN Mazes, Moving MNIST) across four metrics (SSIM, PSNR, LPIPS, FVD), CW-VAE achieves the best average rank (2.25) against VTA, RSSM, SVG-LP, and a no-temporal-abstraction ablation. Headline long-horizon results: accurate Minecraft prediction for over 400 frames where prior work fails before or around 150, and Moving MNIST digit identities remembered across all 1000 prediction steps where all baselines forget before or around step 300. The paper also proposes MineRL Navigate (500-frame sequences) as a long-term video prediction benchmark.

## Context & motivation

Video prediction over long horizons requires learning long-term dependencies, and the dominant approaches were poorly suited to it. Temporally autoregressive models (predicting frame-by-frame in image space, e.g. SVG-LP, SV2P, Video Transformers) are computationally expensive because they operate in high-dimensional pixel space at the dataset's frame rate, and feeding generated frames back into the model accumulates errors and induces distributional shift relative to training. Latent dynamics models (Kalman filters through deep variants like RSSM) predict a learned latent sequence forward and decode it, never feeding generated images back — but standard versions use a single level ticking at the input frame rate. Hierarchical latent variable models (Ladder VAE, VLAE, NVAE, very deep VAEs; HVRNN and Video Flow for video) separate levels of abstraction but still operate at input frequency, which makes very long-range dependencies hard to retain. Temporally abstract models existed but had gaps: Clockwork RNNs (Koutník et al., 2014 — the paper's stated inspiration) partition deterministic RNN units by clock speed without latent variables; VTA (Kim et al., 2019) uses two latent levels where the fast level learns *when* the slow level ticks, but resets its low level whenever the high level steps; TD-VAE makes jumpy predictions without a hierarchy.

CW-VAE's contribution is deliberately simple: combine hierarchical latent dynamics (built on RSSM) with *fixed* clock speeds per level, evaluate extensively (4 datasets, up to 1000-frame rollouts), propose a long-horizon Minecraft benchmark, and analyze *where* the hierarchy stores information (content separation, adaptation to sequence speed, KL per level).

## Method

### Problem formulation

Given a video $x_{1:T}$ (frames $x_t$), learn a generative model that can be conditioned on context frames and then rolled out open-loop in latent space to predict future frames. The model is a hierarchy of $L$ recurrent latent sequences; $s_t^l$ denotes the latent state at timestep $t$ and level $l$. Training maximizes an evidence lower bound (ELBO) on the data likelihood, since the exact likelihood of this latent variable model is not computable in closed form.

### Core idea

All levels of a latent hierarchy transition at different **fixed** clock speeds, slowing down exponentially with height: level $l$ ticks every $k^{l-1}$ steps. Slow levels pay their KL "information cost" rarely, so the ELBO itself pressures the model to store slowly changing global information high up and fast-changing detail low down — no learned segmentation mechanism (as in VTA) is needed.

### Architecture / algorithm

**Clock structure.** The set of *active timesteps* at level $l \in [1, L]$ — the steps at which that level's transition actually generates a new state — is

$$\mathcal{T}_l \doteq \{\, t \in [1, T] \mid t \bmod k^{l-1} = 1 \,\}$$

where $k$ is the temporal abstraction factor (e.g. with $k=2$: $\mathcal{T}_1 = \{1,2,3,\dots\}$, $\mathcal{T}_2 = \{1,3,5,\dots\}$). Each active latent at level $l+1$ conditions the $k$ consecutive latents beneath it at level $l$. Equivalently, one can think of every level as having a state at every timestep, but for inactive steps the state is just copied forward:

$$s_t^l \doteq s^l_{\max\{\tau \in \mathcal{T}_l \mid \tau \le t\}} \quad \forall\, t \notin \mathcal{T}_l,$$

i.e. the state at an inactive step equals the state from that level's most recent active step.

**Joint distribution.** The generative model factorizes into per-frame reconstruction from the *lowest* level and state transitions at every level, each conditioned on the previous state at the same level and the current state one level above:

$$p(x_{1:T}, s_{1:T}^{1:L}) = \Big(\prod_{t=1}^{T} p(x_t \mid s_t^1)\Big) \Big(\prod_{l=1}^{L} \prod_{t \in \mathcal{T}_l} p(s_t^l \mid s_{t-1}^l, s_t^{l+1})\Big).$$

Here $p(x_t \mid s_t^1)$ is the image decoder (only level 1 renders pixels), and $p(s_t^l \mid s_{t-1}^l, s_t^{l+1})$ is the prior transition at level $l$ — temporal context comes from $s_{t-1}^l$ (the same level's previous state, which by the copy convention is its last active state) and top-down context from $s_t^{l+1}$.

**Components.** For all $l \in [1,L]$ and $t \in \mathcal{T}_l$:

- Encoder: $e_t^l = e(x_{t:t+k^{l-1}-1})$ — the embedding feeding an active state at level $l$ summarizes the block of $k^{l-1}$ frames that state is responsible for. Frames are embedded by a CNN; per appendix C, the $k^{l-1}$ frame embeddings are pre-processed by a feed-forward network and **summed** into one embedding.
- Posterior transition: $q(s_t^l \mid s_{t-1}^l, s_t^{l+1}, e_t^l)$ — like the prior but additionally sees the image evidence $e_t^l$. Used at training time and to encode context frames.
- Prior transition: $p(s_t^l \mid s_{t-1}^l, s_t^{l+1})$ — no image input; this is what enables open-loop latent rollout at prediction time. All weights are shared with the posterior except the output layer that predicts mean and variance.
- Decoder: $p(x_t \mid s_t^1)$ — transposed CNN from the lowest-level state. No encoder–decoder skip connections, so nothing bypasses the latents.

**State structure (RSSM-style).** Each state $s_t^l$ splits into a deterministic part $h_t^l$ and a stochastic part $z_t^l$ (following RSSM, Hafner et al. 2019b). The deterministic variable — updated by one GRU per level at every active step — aggregates the temporal and top-down context and then conditions the stochastic variable. The stochastic variables are diagonal Gaussians with predicted means and variances; the deterministic path preserves information over many steps without being erased by noise, while the stochastic path models multiple possible futures.

**Training objective (ELBO, term by term).**

$$\max_{e,q,p}\; \sum_{t=1}^{T} \mathbb{E}_{q_t^1}\big[\ln p(x_t \mid s_t^1)\big] \;-\; \sum_{l=1}^{L} \sum_{t \in \mathcal{T}_l} \mathbb{E}_{q_{t-1}^l q_t^{l+1}}\big[\mathrm{KL}[\,q_t^l \,\|\, p_t^l\,]\big].$$

- The first sum is the **reconstruction term**: at every timestep, the log-likelihood of the true frame under the decoder, with the lowest-level state sampled from the posterior $q_t^1$. Only level 1 has a reconstruction term because only level 1 decodes to pixels.
- The second sum is a **KL regularizer per level, summed over that level's active timesteps only**, with expectation over the posterior samples it is conditioned on (previous same-level posterior $q_{t-1}^l$ and the posterior above $q_t^{l+1}$). Each KL term limits how much information about the images enters that state via the encoder: the model gets the "free" information carried in from the previous latent and the level above without KL cost, and pays only for what it newly extracts from pixels.
- **Why the clock speeds interact with the KL:** higher levels have exponentially fewer active steps, hence exponentially fewer KL terms. Storing slowly-changing information high up is therefore cheaper than either (a) repeatedly re-extracting it from frames at a low level (paying KL every step) or (b) trying to carry it along the fast level for many transitions and risking forgetting. This is the mechanism that produces the observed separation of slow vs fast content — it falls out of the objective, not from any explicit disentanglement loss.

All components jointly optimize this objective with stochastic backpropagation and reparameterized sampling (Kingma & Welling 2013; Rezende et al. 2014).

**Relation to RSSM (the 1-level special case).** RSSM factorizes $p(x_{1:T}, s_{1:T}) = \prod_t p(x_t \mid s_t)\, p(s_t \mid s_{t-1})$ with ELBO $\sum_t \mathbb{E}_{q_t}[\ln p(x_t \mid s_t)] - \sum_t \mathbb{E}_{q_{t-1}}[\mathrm{KL}[q_t \| p_t]]$ — CW-VAE generalizes this with the level index, top-down conditioning, and per-level clocks.

### Derivations / why it works

The paper states the ELBO directly rather than deriving it step-by-step (it is the standard variational bound applied to the factorization above); the load-bearing reasoning is the KL-cost argument reproduced above. _No further derivation is given in the source._

### Training procedure

- All models trained 300 epochs on training sequences of 100 frames at 64×64 pixels.
- Optimizer: Adam, learning rate $3 \times 10^{-4}$, $\epsilon = 10^{-4}$, batch of 100 sequences × 100 frames.
- Default temporal abstraction factor $k = 6$ per level; 3 levels typical (up to 4 levels, and factors up to 8, in the analysis experiments).
- Sizes: encoder output $|e_t^l| = 1024$, stochastic state $|z| = 20$, deterministic state $|h| = 200$, hidden layers in the transition cell 200. For MineRL Navigate these grow to $|z| = 100$, $|h| = 800$, hidden 800.
- Encoder/decoder are DCGAN-discriminator/generator-like CNNs; no skip connections.
- Cost: a 3-level, factor-6 CW-VAE trains in 2.5 days on one Nvidia Titan Xp (20 h per 100 epochs); higher abstraction factors train *faster* since fewer transitions are computed. Parameters: 12M (small) / 34M (large) for 3-level CW-VAE and NoTmpAbs, vs RSSM 5M/13M, SVG 13M/23M, VTA 3M.
- Baseline tuning: learning rate in $[10^{-4}, 10^{-3}]$, decoder stddev in $[0.1, 1]$.

### Inference / sampling

Open-loop video prediction: the model receives 36 context frames — chosen as exactly one step of the slowest level (3 levels × factor 6 → $6^2 = 36$ frames per top-level state), the minimum for the top level to transition once — encoded through the posterior. It then rolls forward using only the prior transitions at every level (no intermediate frames observed, no generated frames fed back), decoding the lowest-level states to pixels.

## Experimental setup

- **Datasets:** MineRL Navigate (~750k frames, crowd-sourced Minecraft navigation processed into a long-horizon benchmark, 500-frame evaluation); KTH Action (290k frames, 600 videos of six human actions); GQN Mazes (9M frames, scripted traversal of procedurally generated mazes with randomized wall/floor textures); Moving MNIST (2M generated frames, two digits with velocities uniform in 2–6 px/frame, bouncing).
- **Baselines:** VTA (state-of-the-art temporally abstract latent model), RSSM (single-level latent dynamics), SVG-LP (image-autoregressive with learned prior), and **NoTmpAbs** (CW-VAE ablation with abstraction factor 1 at all levels — same parameter count, hierarchy without temporal abstraction).
- **Metrics:** SSIM (↑), PSNR (↑), LPIPS (↓), FVD (↓), averaged across frames of all evaluation sequences; aggregate = average rank across all dataset×metric cells.

## Key results

- **Average rank (lower better):** CW-VAE 2.25 > NoTmpAbs 2.62 > RSSM 2.75 > SVG 3.31 > VTA 3.62. Notably, temporal abstraction contributes more than hierarchy alone: NoTmpAbs is sometimes beaten by RSSM, while CW-VAE matches or beats RSSM on every dataset and metric.
- Sample numbers (Table 1): MineRL — CW-VAE 0.65 SSIM / 23.23 PSNR / 0.31 LPIPS / 2612 FVD (vs VTA 0.56/18.50/0.35/3305). KTH — CW-VAE 0.85 SSIM / 26.52 PSNR. Moving MNIST — CW-VAE 593 FVD vs SVG 1392. SVG wins parts of GQN Mazes (0.55 SSIM, 820 FVD) on sharpness.
- **MineRL Navigate (500 frames):** CW-VAE stays accurate for over 400 frames (predicts the approach to an island and diverse terrain); prior work fails before or around 150. VTA shows sky artifacts after ~240 steps; SVG essentially copies the initial frame.
- **Moving MNIST (1000 frames):** CW-VAE retains digit identities for all 1000 steps; all baselines forget before or around step 300, SVG within ~50 — supporting the claim that image-space feedback accumulates error faster than latent-space rollout. Positions are accurate to ~100 steps, plausible after.
- **KTH:** accurate walking motion for ~50 steps vs ~20 for RSSM/SVG; CW-VAE uses the smaller DCGAN encoder where SVG needed VGG.
- **GQN Mazes:** CW-VAE maintains wall/floor textures of both rooms across 200 frames; RSSM and VTA fail to predict the second room's textures; SVG is sharpest but mixes textures over long horizons.
- Reported failure mode: latent-space models' predictions are slightly blurry vs SVG; MineRL predictions blurry overall (attributed to restricted model capacity).

## Ablations

- **Temporal abstraction factor (Moving MNIST, factors 2/4/6/8, equal parameters):** larger $k$ directly extends the horizon over which predictions stay accurate; CW-VAE holds temporal context ~6× longer than RSSM/SVG before degrading to the "random training frame" baseline level.
- **KL per level vs depth (Table 2, Moving MNIST):** summed KL at level 1 drops as depth grows (440.5 for 1-level → 397.1 for 4-level), with information redistributed upward (4-level: 397.10 / 41.20 / 2.66 / 0.0001 nats at levels 1–4). Higher levels store less — the dataset has more short-term than long-term structure; the top level plausibly only needs the two digit identities (~5 nats, authors' conjecture).
- **Content separation (GQN Mazes):** rolling out with the top level sampled from its *prior* (blind to context) while lower levels see context → camera and nearby wall positions are preserved (stored low) but textures randomize (stored top). Resetting lower vs middle levels showed similar content, suggesting 2 levels may suffice for this dataset.
- **Adaptation to sequence speed:** on faster/slower Moving MNIST variants, per-level KL tracks frame rate — faster digits push information into the fast bottom level, slower digits shift it into middle/top levels. The hierarchy allocates information by timescale automatically.
- **NoTmpAbs** (factor 1) is itself the hierarchy-without-clocks ablation; see benchmark ranks above.

## Limitations

Paper's own (all acknowledged in §4): (1) standard video metrics poorly capture long-horizon quality — best-of-100-samples evaluation didn't change conclusions, and they suggest datasets with ground-truth scene attributes plus readout networks instead; (2) with 100-frame training sequences, 3 levels and factor 8, the top level transitions only once per training sequence, so it is barely trained on consecutive transitions; (3) small DCGAN-scale encoders/decoders limit image quality (blurriness), and larger architectures are conjectured to help. Honest-reader additions [analyst's view]: fixed clock speeds are a hand-set hyperparameter, not adaptive (VTA's learned ticking addresses this, though worse empirically here); all datasets are 64×64 and visually simple by later standards; evaluation is on stochastic-sample quality rather than downstream control, despite the world-model motivation.

## Why it matters [analyst's view]

This is the cleanest early demonstration that *temporal* abstraction in latent dynamics — not just representational hierarchy — is what buys long-horizon video prediction, with the elegant mechanism that per-level KL pricing sorts information by timescale without any explicit objective for it. Two of its bets are now mainstream in the vault's 2026 material. First, *predict in latent space, never feed pixels back*: that is exactly the design of [[papers/baldassarre-2025-dino-world-models]] (rollouts in frozen DINOv2 feature space) and [[papers/kerssies-2026-delta-tokens]] (1-D sequences of feature-space delta tokens); DeltaTok's compression of *what changed between frames* is a close cousin of CW-VAE's intuition that most frame content is redundant and only the residual should cost bits — implemented as tokenization rather than KL pricing. Second, *the horizon problem*: CW-VAE's "accurate to 400+ frames where others die at 150" framing anticipates the fidelity–horizon gap that [[papers/cai-2026-mode-mean-seeking]] attacks for minute-scale generation — though the modern solution (decoupled diffusion heads, teacher distillation) abandons hierarchical clocks entirely. In the taxonomy of [[papers/ding-2024-world-models-survey]], CW-VAE sits in the direct lineage from Ha & Schmidhuber and RSSM/PlaNet/Dreamer toward video-generation world models. Interestingly, explicit multi-rate latent hierarchies mostly *disappeared* from the strongest recent systems (attention over long contexts took their place), which makes CW-VAE a good test case for asking what was lost: none of the recent notes report the clean slow/fast content separation this paper demonstrates, and its KL-per-level probe remains a nice interpretability tool for any stochastic latent world model.

## Open questions / things to verify

- Would training on longer sequences (so the top level sees many transitions) substantially improve the slow levels, as the authors conjecture?
- Does the timescale-sorting mechanism survive when the reconstruction loss is replaced by feature-space losses (DINO-style), or does it depend on pixel-level KL economics?
- How would fixed clocks compare against learned ticking (VTA-style) with modern capacity — is VTA's weakness here fundamental (its low-level reset) or an artifact of scale?
- The Minecraft benchmark: did later long-horizon work adopt it? _not addressed by the source_ — worth checking against current long-video benchmarks (e.g. VBench-Long used in [[papers/cai-2026-mode-mean-seeking]]).
- Downstream use for planning/RL is motivated but untested here — the natural experiment is a Dreamer-style agent on top of a clockwork hierarchy.

## Connections

- Builds on: RSSM / PlaNet (Hafner et al., 2019b) — _needs note_; Clockwork RNNs (Koutník et al., 2014) — _needs note_; VTA (Kim et al., 2019) — _needs note_
- Surveyed within: [[papers/ding-2024-world-models-survey]] (world-model lineage this paper belongs to)
- Contrasts with / anticipated by later latent-space video world models: [[papers/baldassarre-2025-dino-world-models]], [[papers/kerssies-2026-delta-tokens]]
- Long-horizon video generation, modern take on the horizon problem: [[papers/cai-2026-mode-mean-seeking]]
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/generative-models]], [[topics/recurrence]]
- Author indices: [[authors/danijar-hafner]], [[authors/vaibhav-saxena]], [[authors/jimmy-ba]] _(create only if/when these authors accumulate more vault items — singleton discipline)_

## Selected quotes

> "The KL regularizers limit the amount of information about the images that enters via the encoder. ... Hence it is easier for the model to store slowly changing information high up in the hierarchy than to pay a KL penalty to repeatedly extracting the information from the images at the lower level or trying to remember it by passing it along for many steps at the lower level without accidental forgetting." — §2, Training objective

> "On the Minecraft Navigate dataset, CW-VAE accurately predicts for over 400 frames, whereas prior work fails before or around 150 frames." — §1, key contributions

> "We observe that CW-VAE remembers the digit identities across all 1000 frames of the prediction horizon, whereas all other models forget the digit identities before or around 300 steps." — Figure 6 caption

> "The faster the digits move, the more the fast ticking lowest level of the hierarchy is used. The slower the digits move, the more the middle and top level are used." — §3.9
