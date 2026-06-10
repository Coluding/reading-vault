---
type: paper
title: "Looped Diffusion Language Models"
authors: ["Sanghyun Lee", "Chunsan Hong", "Seungryong Kim", "Jonghyun Lee", "Jongho Park", "Dongmin Park"]
year: 2026
venue: "arXiv:2605.26106"
url: https://arxiv.org/abs/2605.26106
rw_id: 01kt7ndspbccq5ar718xm6k09r
topics: [looped-transformers, recursive-reasoning, generative-models]
priority: high
read_state: skimmed
relevance: ""
added: 2026-06-03
last_updated: 2026-06-10
---

# Looped Diffusion Language Models (LoopMDM)

## TL;DR

LoopMDM is the **first application of looped transformers to masked diffusion
language models (MDMs)**. Instead of looping the whole network, it *selectively*
loops a small set of **early-middle layers** of the denoiser, sharing their
weights across $S$ repetitions. This buys a depth-scaling effect at **zero extra
parameters**, and the loop count $S$ becomes an **inference-time compute knob**.
Under matched total training FLOPs it reaches the same test NLL as an equal-size
MDM with up to **3.3× fewer training FLOPs**, beats it on reasoning (**+8.5 on
GSM8K**), and even surpasses *deeper* non-looped MDMs with comparable per-step
compute — so the gain is from repeated shared computation, not just more depth.

## Context & motivation

MDMs (LLaDA, Dream, Seed Diffusion) are a credible non-autoregressive alternative
to ARMs, but progress has mostly come from more params / more tokens. The looped
transformer — a shared block applied repeatedly — is a known AR trick for
test-time compute scaling, but **had not been tried in MDMs**. It's *a priori*
unclear it should help, since MDMs already iterate across denoising steps. Prior
looping work focused on parameter efficiency or inference-time refinement, and
gains at *fixed training compute* were non-trivial. LoopMDM asks: integrate
looping into MDMs *under matched training compute* — where and how much to loop?

## Background: the masked-diffusion framework (§2.1)

*The paper's preliminaries. This is the shared MDM scaffolding (LLaDA / Dream /
MDLM / MD4 lineage) that LoopMDM plugs its looped denoiser into — worth keeping
in full because every result below is stated against these three equations.*

### Notation
Sequences $x = (x^1,\dots,x^L)$ of length $L$; each token $x^i$ is a one-hot over
vocabulary $V=\{1,\dots,V\}$. Augment with a **mask token** $m$: $\bar V = V\cup\{m\}$.
$\mathrm{Cat}(\cdot\mid p)$ is a categorical with probability vector $p$. The one
piece of notation that unlocks everything is the **point mass** $\delta_a(b)$ — a
distribution that puts *all* its probability on the single value $a$, so as a
function of the outcome $b$ it is just the indicator "$1$ if $b=a$, else $0$"
(the discrete Dirac / Kronecker delta). Every equation below is a **mixture of
point masses**: deltas are the building blocks ("definitely this value"), the
$\alpha$'s are the mixing weights.

### Forward process (corruption) — Eq. (1)
A strictly decreasing schedule $\alpha_t:[0,1]\to[0,1]$ runs from $\alpha_0\simeq1$
(clean) to $\alpha_1\simeq0$ (all masked). Each token is corrupted **independently**:

$$q(x^i_t \mid x^i_0) = \alpha_t\,\delta_{x^i_0}(x^i_t) + (1-\alpha_t)\,\delta_m(x^i_t)$$

A per-token coin flip: **keep** the clean token $x^i_0$ with prob $\alpha_t$,
**replace with mask** $m$ with prob $1-\alpha_t$. Because the only two point
masses in the mixture are $\delta_{x^i_0}$ and $\delta_m$, a token can *only* end
up unchanged or masked — never corrupted into a different real token (this is
what "absorbing-state" means, and what separates masked diffusion from Gaussian
diffusion). Reading the formula at each outcome: $x^i_t=x^i_0\to\alpha_t$,
$x^i_t=m\to 1-\alpha_t$, anything else $\to 0$. So $1-\alpha_t$ is literally the
expected fraction of masked positions at time $t$.

> `The cat sat on the mat` → (at some $t$) → `The [M] sat on [M] mat`

### Reverse process (denoising) — Eq. (2)
Generation runs the schedule backwards: start from the all-mask sequence
$x_1=(m,\dots,m)$ and progressively unmask, $x_1\to x_t\to x_s\to\cdots\to x_0$
with $s<t$. A denoiser $p_\theta(x^i_0\mid x_t)$ reads the whole partially-masked
sequence and predicts a distribution over the **clean** token at each masked
position. With $\alpha_{s\mid t}=(1-\alpha_s)/(1-\alpha_t)$, masked positions update as

$$p_\theta(x^i_s\mid x_t) = \alpha_{s\mid t}\,\delta_m(x^i_t) + (1-\alpha_{s\mid t})\,\mathrm{Cat}\!\big(x^i_s\mid p_\theta(x^i_0\mid x_t)\big),\qquad x^i_t=m.$$

Same mixture-of-point-masses shape as the forward process, read in reverse. For a
currently-masked position: **stay masked** with prob $\alpha_{s\mid t}$ (the
$\delta_m$ term), or **reveal** with prob $1-\alpha_{s\mid t}$, sampling the actual
token from the model's predicted clean distribution. Two structural facts:
- **Already-unmasked tokens are frozen** — the transition only acts on masked
  positions; once revealed, a token never changes again.
- $\alpha_{s\mid t}$ compares masking fractions: $1-\alpha_t$ masked now vs.
  $1-\alpha_s$ masked at the (less noisy) target $s$. Since fewer are masked at
  $s$, the ratio $<1$ and each step reveals a fraction of the remaining masks.

> `The [M] sat on [M] mat`, model predicts pos-2 {cat .8, dog .1, …} & pos-5
> {mat .7, floor .2, …}; with $\alpha_{s\mid t}=0.4$, each mask reveals w.p. 0.6 →
> `The cat sat on [M] mat` → … → `The cat sat on the mat`.

**Conditional independence:** the reverse process factorizes,
$p_\theta(x_s\mid x_t)=\prod_i p_\theta(x^i_s\mid x_t)$ — once the transformer has
produced its predictions, masked positions are sampled independently. All the
cross-position dependence lives in the transformer's contextual representation,
not in the sampling step.

### Training objective (NELBO) — Eq. (3)
$$\mathcal{L}_{\text{NELBO}} = \mathbb{E}_{t\sim U[0,1],\,x_t\sim q}\!\left[\sum_{i=1}^L \mathbb{I}[x^i_t=m]\,\frac{\alpha_t'}{1-\alpha_t}\,\log\langle p^i_\theta(x_t,t),\,x^i_0\rangle\right]$$

Three pieces:
- **$\mathbb{I}[x^i_t=m]$** — only **currently-masked** positions contribute. The
  model is graded purely on reconstruction, exactly like BERT-style masked LM.
- **$\langle p^i_\theta,\,x^i_0\rangle$** — inner product of the predicted
  distribution with the one-hot true token simply **selects the probability mass
  on the correct token**, so $\log\langle\cdot,\cdot\rangle$ is the standard
  cross-entropy / log-likelihood of the right token.
- **$\frac{\alpha_t'}{1-\alpha_t}$** — the schedule-derived per-timestep weight
  from the ELBO derivation ($\alpha_t'<0$ since $\alpha_t$ decreases; that sign
  flips the bound so *minimizing* the NELBO *maximizes* weighted log-likelihood).
  This factor is the only thing separating a valid diffusion ELBO from a plain
  masked-LM loss — strip it and you're back to BERT.

[analyst] Net: sample a noise level, mask tokens by Eq. (1), train the network to
predict the originals at masked spots via weighted cross-entropy. The whole
generative story is "predict the masked tokens, commit some, repeat." LoopMDM
changes *only* the denoiser $p_\theta$ (looping its early-middle layers) — the
forward process, the reverse-transition form, and this objective are inherited
unchanged, which is why its NELBO (above) is identical save the loop-count
superscript $S$.

## Method

### Problem formulation
Standard masked diffusion. Forward process corrupts each token to a mask with
prob $1-\alpha_t$; the denoiser $p_\theta(x_0^i \mid x_t)$ predicts clean tokens
at masked positions; trained with the MDM NELBO:

$$\mathcal{L}_{\text{NELBO}} = \mathbb{E}_{t,S,x_t}\!\left[\sum_{i=1}^L \mathbb{I}[x_t^i=\mathbf{m}]\,\frac{\alpha_t'}{1-\alpha_t}\log\langle p_\theta^{S,i}(x_t,t), x_0^i\rangle\right]$$

### Core idea
Split the denoising transformer into **head** ($n_h$, run once), a **looped
mid-block** ($n_m$, weights shared across $S$ applications), and **tail** ($n_o$
+ LM head, run once). Forward pass:

$$H_t^{(0)} = \mathtt{head}(x_t),\quad H_t^{(k)} = \mathtt{mid}(H_t^{(k-1)}),\quad p_\theta^S = \mathtt{tail}(H_t^{(S)})$$

Effective depth $n_h + S\cdot n_m + n_o$; parameter count stays $n_h+n_m+n_o$.
Head/tail kept unshared to preserve embedding/projection specialization.

### What to loop (their key finding)
Three knobs: number of looped layers $n_m$, **position** of the looped block, and
max training loop $S_{\max}$. Findings: (1) **few layers suffice** — looping more
adds cost without benefit; (2) **early-middle placement is best** (loop after
low-level token reps form, before late layers specialize for prediction).

### Training with stochastic loop count
Sample $S \sim \mathcal{U}\{1,\dots,S_{\max}\}$ per step, applying NELBO to the
final loop output. This exposes the shared block to many effective depths, gives
stable reps across $S$, and **enables any $S$ at inference, including $S > S_{\max}$**.
Per-step FLOPs: $F_\text{loop} = F_\text{base} + (\mathbb{E}[S]-1)\cdot n_m\cdot F_\text{layer}$
with $\mathbb{E}[S]=(S_{\max}+1)/2$. Fair comparison via **matched total training
FLOPs** (train LoopMDM for proportionally fewer steps).

### Training procedure
170M-param diffusion transformer (RoPE + adaLN), 12 layers; looping mid-layers
**1–2** (zero-indexed) with $S_{\max}=12$. Corpora: OpenWebText, LM1B,
FineWeb-Edu. Baseline MDM: 1M steps, batch 512, sequence packing. Seq len 128
(LM1B) / 1024 (OWT, FineWeb-Edu).

## Experimental setup

- **Datasets:** FineWeb-Edu, OpenWebText, LM1B (pretraining); 7 external corpora for zero-shot PPL; 9 downstream benchmarks (ARC, BoolQ, HellaSwag, PIQA, RACE, SIQA, WinoGrande, OBQA); GSM8K + Sudoku for reasoning.
- **Baseline:** equal-param, equal-FLOPs non-looped MDM; also deeper non-looped MDMs at matched per-step compute.
- **Metrics:** test NLL, generative perplexity, zero-shot perplexity, downstream accuracy.

## Key results

- **Training efficiency:** for $S\geq 6$ LoopMDM matches baseline test NLL with **3.34× / 2.95× / 2.34×** fewer training FLOPs on LM1B / OWT / FineWeb-Edu (Fig. 2).
- **Beyond depth:** **+8.5 on GSM8K** over equal-size baseline, and beats deeper MDMs with matched per-step FLOPs — gain is from repeated shared computation, concentrated in math.
- **Extrapolation:** $S=1$ *underperforms* baseline (shared block needs repetition); $S=24 > S_{\max}=12$ keeps improving — the learned operator generalizes past its training range.
- **Generative perplexity** (Table 1) drops monotonically with $S$; gains largest at *small* denoising-step budgets (e.g. 64 steps: 116.7 → 99.1 at $S=12$).
- **Zero-shot PPL** (Table 2): improves on all 7 corpora at $S=12$ (PTB 108.5 → 90.2); most gains already at $S=6$.
- **Downstream** (Table 3): outperforms MDM on most of 9 benchmarks as $S$ grows.

## Ablations / analysis

- **Where/how much to loop** (§4.1): early-middle, few layers — the core selective-design ablation.
- **Sudoku with restricted generation order**: looping lets the model solve a globally-constrained task a shallow baseline cannot — evidence looping adds genuine iterative reasoning.
- **Attention analysis**: **mask-to-mask interactions increase with loop count** — the proposed mechanism for why looping helps MDMs specifically.
- **Timestep study**: gains concentrated at **intermediate noise levels**, motivating an **adaptive inference** strategy that allocates more loops where they help, cutting compute while holding performance.

## Limitations

[analyst] Demonstrated at 170M scale only — whether the 3.3× efficiency and
GSM8K gains hold at LLaDA/Dream scale (billions of params) is the obvious open
question. Per-step compute rises with $S$, so the "free depth" is paid for at
inference; the win is the *training*-FLOP efficiency and the optional inference
knob. The selective-loop config (layers 1–2, $S_{\max}=12$) is tuned and may not
transfer across architectures.

## Why it matters [analyst's view]

This is the bridge between two vault threads: **looped/recurrent computation**
([[topics/looped-transformers]], [[topics/recursive-reasoning]]) and
**diffusion/MDM generative modeling** ([[topics/generative-models]]). It's a neat
existence proof that looping helps *even when the model already iterates* (across
denoising steps) — the loop adds intra-step refinement that the denoising chain
doesn't provide, specifically more mask-to-mask interaction. The
"stochastic-$S$-during-training → arbitrary-$S$-at-inference" recipe is the same
inference-time-scaling lever seen in the AR looped-transformer papers
([[papers/deng-2026-lt2-looped]], [[papers/chen-2026-training-free-looped]]) and
echoes the latent-recursion line ([[papers/wang-2025-hierarchical-reasoning-model]],
[[papers/baek-2026-gram]]).

## Open questions / things to verify

- Does the efficiency gain survive at billion-parameter MDM scale?
- Is early-middle placement universal, or dataset/scale-dependent?
- How does adaptive-loop inference compare to simply using more denoising steps at equal compute?
- Why does looping help *MDMs* more than the already-iterative denoising chain — is mask-to-mask interaction the full story?

## Connections

- Builds on: looped/Universal Transformers (AR literature)
- Related: [[papers/deng-2026-lt2-looped]], [[papers/chen-2026-training-free-looped]], [[papers/wang-2025-hierarchical-reasoning-model]], [[papers/baek-2026-gram]]
- Contrasts with: [[papers/ding-2024-diffusion-world-model]] (diffusion that *removes* iteration)
- Topic MOCs: [[topics/looped-transformers]], [[topics/recursive-reasoning]], [[topics/generative-models]]
- Author indices: [[authors/sanghyun-lee]], [[authors/dongmin-park]]

## Selected quotes

> "Selectively looping a few early-middle transformer layers captures the full benefit of repeated computation at substantially lower training cost than uniform looping." — §1

> "At $S = 24$, beyond the training maximum $S_{\max} = 12$, performance continues to improve, indicating that the learned operator generalizes beyond the training regime." — §4.1
