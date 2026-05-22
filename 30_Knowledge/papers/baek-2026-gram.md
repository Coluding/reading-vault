---
type: paper
title: "Generative Recursive Reasoning"
authors: ["Junyeob Baek", "Mingyu Jo", "Minsu Kim", "Mengye Ren", "Yoshua Bengio", "Sungjin Ahn"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2605.19376
rw_id: 01ks7dp5jgc81wgnqyr9vq5hjc
topics: [recursive-reasoning, latent-reasoning, variational-inference, inference-time-scaling, generative-models]
priority: high
read_state: queued
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# Generative Recursive Reasoning (GRAM)

**TL;DR** — GRAM extends deterministic Recursive Reasoning Models (HRM, TRM, Looped Transformers) into *probabilistic* multi-trajectory recursion. At each recursion step a Gaussian "stochastic guidance" perturbation $\epsilon_t \sim \mathcal{N}(\mu_\theta, \sigma_\theta^2 I)$ is added to the deterministic update $u_t$, turning the latent trajectory into a sample from a learned distribution. Trained with **amortized variational inference** (ELBO over latent trajectories), GRAM enables (i) parallel-trajectory inference-time scaling, (ii) coverage of multi-solution tasks where HRM collapses, and (iii) an *unconditional* generative interpretation $p_\theta(x)$ via empty conditioning. Headline numbers: 99.7% accuracy on N-Queens 8×8 (vs HRM 78.7%, TRM 66.8%), 97.0% Sudoku-Extreme with N=20 parallel samples at 16 iterations (vs TRM 90.5% at 320 iterations).

## Context & motivation

The paper's framing problem (§1):

> "Recent recursive reasoning models such as HRM and TRM provide early evidence for the potential of this approach in structured reasoning ... Existing RRMs, however, remain fundamentally deterministic: given the same input and initialization, they follow a single latent trajectory and converge to a single prediction. This deterministic recursion collapses the space of plausible reasoning paths into a single attractor."

The conceptual move: future RRMs should be *deep* (repeated refinement, as today) *and wide* (parallel trajectory exploration, as GRAM proposes).

Three explicit contributions:
1. Formulate recursive reasoning as a latent-variable generative process.
2. Width-based inference-time scaling (parallel trajectory sampling).
3. Empirical evidence that the formulation improves structured reasoning, multi-solution constraint satisfaction, and unconditional generation.

## Method (§2)

### Architecture
A fixed initial latent state $z_0$. At each transition step $t$:
$$z_t \sim p_\theta(z_t \mid z_{t-1}, e_x), \quad t = 1, \ldots, T$$
where $e_x = f_\text{enc}(x; \theta)$ is the input embedding reused throughout. After $T$ transitions, decoder produces $\hat{y} = \arg\max f_\text{dec}(z_T; \theta)$. The full computation stacks $N_\text{sup}$ supervision steps, with each $z_T^{(n)}$ feeding into $z_0^{(n+1)}$ — same outer-loop structure as HRM's deep supervision.

### Stochastic latent transitions
The deterministic update $u_t$ is unchanged, but a learned Gaussian perturbation is added:
$$\epsilon_t \sim p_\theta(\epsilon_t \mid u_t) := \mathcal{N}\big(\mu_\theta(u_t), \sigma_\theta^2(u_t) I\big)$$
$$z_t = u_t + \epsilon_t$$

> "The mean $\mu_\theta(u_t)$ encodes a state-dependent direction in which the trajectory is steered, while the variance $\sigma_\theta^2(u_t)$ controls the amount of exploration."

This is **stochastic guidance**: not white noise, but a learned, state-dependent perturbation.

### Hierarchical instantiation (§2.1)
GRAM borrows HRM's two-level state $z = (h, l)$:
- $K$ low-level updates per transition with $h$ frozen:
$$l_{t,k} = f_L(h_{t-1}, l_{t,k-1}, e_x; \theta), \quad k = 1, \ldots, K$$
- Then a stochastic high-level update:
$$u_t = f_H(h_{t-1}, l_t; \theta)$$
$$h_t = u_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(\mu_\theta(u_t), \sigma_\theta^2(u_t) I)$$

**Stochasticity only at the high level**: "the low-level refinement is fully deterministic, while the stochastic guidance signal $\epsilon_t$ acts on the slower, more abstract component of the latent state." The paper notes injecting noise into the low-level state was tried and didn't help.

### Training — Amortized Variational Inference (§2.2)

GRAM is trained by maximising the ELBO with respect to a target-conditioned variational posterior $q_\phi(\tau \mid x, y)$:

$$\log p_\theta(y \mid x) \geq \mathbb{E}_{q_\phi(\tau \mid x, y)}[\log p_\theta(y \mid \tau, x)] - \mathrm{KL}(q_\phi(\tau \mid x, y) \| p_\theta(\tau \mid x))$$

The posterior shares the deterministic transition module with the prior but samples from a target-aware noise distribution. At inference time (no $y$), trajectories are sampled from the prior. Since stochasticity enters only through $\epsilon_{1:T_\text{Total}}$, the trajectory distribution can be represented entirely in noise space.

### Latent Process Reward Model (LPRM)
For inference-time best-of-N selection across parallel trajectories, GRAM trains an LPRM that predicts output correctness from latent state. Used to select the best trajectory when sampling $N$ in parallel.

## Experimental setup (§4)

- **Structured reasoning**: Sudoku-Extreme [from HRM paper], ARC-AGI-1/2.
- **Multi-solution constraint satisfaction**: N-Queens (8×8, 10×10), Graph Coloring (8-vertex, 10-vertex).
- **Unconditional generation**: binarized MNIST, Sudoku-from-empty-board.
- **Baselines**: direct prediction (Transformer 8L / 32L), Looped TF, HRM, TRM, plus generative models (AR, MDLM) for multi-solution tasks. Reproduced under identical settings.

## Key results

### Sudoku-Extreme + ARC-AGI (§4.1, Fig. 3)
- GRAM "consistently outperforms prior recursive models across all benchmarks."
- **Parallel scaling wins big**: "GRAM with $N=20$ samples at 16 iterations outperforms all deterministic baselines at 320 iterations, including TRM (97.0% vs 90.5%), despite comparable computational budget."

### N-Queens / Graph Coloring (§4.2, Table 1)
Headline single-sample accuracy:
| Method | Params | N-Queens 8×8 | N-Queens 10×10 | GC 8-vertex conflict | GC 10-vertex conflict |
|---|---|---|---|---|---|
| Direct Pred (8L) | 27M | 40.4% | 13.6% | 179.3 | 198.7 |
| Looped TF | 7M | 68.4% | 50.0% | 136.0 | 157.3 |
| HRM | 27M | 78.7% | 37.4% | 109.7 | 164.3 |
| TRM | 7M | 66.8% | 17.5% | 109.3 | 170.7 |
| AR | 10.6M | 96.3% | 90.0% | 19.0 | 61.3 |
| MDLM | 12.6M | 96.1% | 74.3% | 2.7 | 12.0 |
| **GRAM** | **10M** | **99.7%** | **89.7%** | **2.7** | **3.3** |

Coverage with 20 samples (% of unique valid solutions discovered): GRAM 90.3% on N-Queens 8×8 vs AR 84.8% vs MDLM 87.2%; vs HRM 26.7%, TRM 36.1%. **Deterministic recursive models structurally cannot capture multiple solutions** — coverage caps at ~36%.

### Unconditional generation (§4.3)
- **Sudoku-from-empty**: 99.05% validity with 10.9M params + 16 supervision steps; D3PM baseline uses 55.1M params + 1000 denoising steps.
- **Binarized MNIST**: IS improves monotonically with recursive depth even beyond training-time (IS 1.85 → 2.04, FID 84.08 → 73.34 as steps go 8 → 256). TRM exhibits mode collapse (FID 303.29).

## Ablations (§4.4)

Architecture ablation (Sudoku-Extreme / N-Queens 8×8):
- Base Looped TF: 61.25 / 71.30
- + DS + HR (= HRM/TRM): 87.40 / 80.70
- + SG (stochastic guidance only): 65.64 / 86.30
- + DS + SG: 73.90 / **100.00**
- + DS + HR + SG (= GRAM): **93.96 / 99.69**

Mechanism ablation:
- w/o stochastic guidance: 82.87 / 72.91 (loses N-Queens)
- stochasticity only (zero mean, variance only): 94.88 / 50.27 (Sudoku OK, N-Queens collapses)
- guide only (variance = 0): 0.00 / 0.00 (catastrophic)
- direct prediction (no recursion): 63.43 / 61.44
- TRM + stochastic decoder: 82.87 / 71.66 (no real gain)
- TRM + random init $z_0 \sim \mathcal{N}(0, I)$: 78.53 / 71.82 (no real gain)

> "Neither improves performance, demonstrating that GRAM's gains stem from the variational framework rather than mere randomness."

## Limitations

Stated (§5):
- "The sequential nature of deep supervision limits training efficiency compared to Transformers, posing a significant barrier to scaling GRAM toward larger foundation models."

[analyst's view]
- All evaluation tasks are structured-reasoning puzzles. Behaviour on natural-language reasoning is untested.
- LPRM for inference-time best-of-N adds a separate trained component; ablations on the LPRM itself aren't shown in the fetched body.
- The "width-based scaling" trades latency for inference cost. Whether it's actually cheaper per accuracy point than just running TRM longer depends on the constant factors of $f_L / f_H$.

## Why it matters [analyst's view]

GRAM is the *right* generalisation of HRM. Three reasons:

1. **It cleanly fixes HRM's coverage problem.** HRM's 26.7% coverage on N-Queens isn't a tuning issue — it's structural. Deterministic recursion can't produce multiple solutions from the same input. GRAM doesn't bolt on a sampling trick; it puts a learned variational posterior at the architectural level. The ablation that *random* perturbations don't help (TRM + random init) is the strongest evidence: the gain comes specifically from learning where to add noise, not from noise per se.

2. **Width-based inference-time scaling is a new axis.** Most "test-time compute" stories (CoT chains, tree-of-thought, MCTS) scale along generation length or search width *in token space*. GRAM scales in *latent* trajectory space — sample $N$ stochastic recursions, select with LPRM. Same computational genre as Monte Carlo rollouts in RL, but inside a feed-forward latent computation.

3. **The unified $p(y \mid x)$ and $p(x)$ story is genuinely interesting.** GRAM trained with target-conditioned posterior becomes an unconditional generator just by dropping the conditioning. This is the standard latent-variable trick (VAE: same encoder for $p(x)$ and $p(x \mid c)$), but applied to *reasoning architectures*. If this generalises, future "reasoning" models double as "generative" models. The Sudoku-from-empty result (99.05% valid boards with 10.9M params) is the most concrete evidence.

GRAM also explicitly references TRM (Tiny Recursive Models) as a deterministic baseline — TRM is not yet in vault, worth follow-up. And the same authors (Sungjin Ahn et al., KAIST/Mila/NYU) have a clear research line stitching together recursive reasoning + generative modelling.

## Open questions

[analyst's view]
- **How is LPRM trained?** Reward model design is half the story for best-of-N inference. Not unpacked in the fetched body.
- **Why does naive stochasticity (TRM + random $z_0$) not help, but learned $\mu_\theta$ does?** The ablation suggests the *direction* of perturbation matters more than the noise. What does $\mu_\theta(u_t)$ actually learn? An interpretability follow-up here would be revealing.
- **Does the unconditional generator have likelihood-evaluation capabilities?** Sudoku-validity is a proxy; PLL or perplexity on natural-text would be a much harder test.
- **Stochastic guidance under sequence length.** All experiments are on small structured tasks. Whether the same recipe handles longer reasoning traces (e.g., math olympiad) is the obvious scale-up question.
- **Relationship to diffusion-language models** like [[papers/yang-2026-replaid-continuous-diffusion]] — both use latent Gaussian transitions with a learned schedule, but the analogue between "denoising steps" and "supervision steps" hasn't been formally drawn out.

## Connections

- **Extends**: [[papers/wang-2025-hierarchical-reasoning-model]] — same hierarchical $h/l$ instantiation; adds stochastic high-level updates and variational training.
- **Direct contrast**: deterministic RRM baselines (HRM, TRM, Looped Transformers).
- **Cousin / mechanism overlap**: [[papers/yang-2026-replaid-continuous-diffusion]] — both add Gaussian noise to latent dynamics with learned scaling; different ends (reasoning vs language modelling).
- **Cousin / framework**: [[papers/maes-2026-leworldmodel]] — both are recurrent latent-computation systems trained without BPTT-as-default (LeWM uses SIGReg; GRAM uses ELBO + deep supervision).
- **Topic MOCs**: [[topics/recursive-reasoning]], [[topics/latent-reasoning]]
- **Author indices**: [[authors/junyeob-baek]]

## Selected quotes

> "Recursive Reasoning Models (RRMs) offer a promising alternative to autoregressive sequence extension by performing iterative latent-state refinement with shared transition functions. Yet existing RRMs are largely deterministic, following a single latent trajectory and converging to a single prediction." — abstract

> "GRAM with $N=20$ samples at 16 iterations outperforms all deterministic baselines at 320 iterations, including TRM (97.0% vs 90.5%), despite comparable computational budget." — §4.1

> "We test two simple approaches to add stochasticity to TRM: (1) stochastic decoding, which samples from the output distribution instead of argmax, and (2) random initialization ... Neither improves performance, demonstrating that GRAM's gains stem from the variational framework rather than mere randomness." — §4.4
