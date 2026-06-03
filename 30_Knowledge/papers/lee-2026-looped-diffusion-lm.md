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
last_updated: 2026-06-03
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
