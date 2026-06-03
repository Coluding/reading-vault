---
type: paper
title: "LT2: Linear-Time Looped Transformers"
authors: ["Chunyuan Deng", "Yizhe Zhang", "Rui-jie Zhu", "Yuanyuan Xu", "Jiarui Liu", "T. S. Eugene Ng", "Hanjie Chen"]
year: 2026
venue: "arXiv:2605.20670"
url: https://arxiv.org/abs/2605.20670
rw_id: 01kt7newbhhj0erxyeyhkrd28g
topics: [looped-transformers, recursive-reasoning, efficient-attention]
priority: high
read_state: skimmed
relevance: ""
added: 2026-06-03
last_updated: 2026-06-03
---

# LT2: Linear-Time Looped Transformers

## TL;DR

Looped transformers (LT) get effective depth by re-applying shared layers $T$
times, but pairing that with full softmax attention re-pays the **quadratic** cost
on every loop — attention becomes the scaling bottleneck. LT2 swaps softmax
attention for **subquadratic** token mixers: **LT2-linear** (linear attention:
GDN, KDA, Mamba2…) and **LT2-sparse** (sliding-window / NSA / DSA). The key
finding is *synergy*: looping a DPLR linear-attention block turns its rank-1 state
update into a **rank-$T$** update, and looping a window-$w$ sparse block expands
the receptive field to **$O(Tw)$** — "looping turns compute into context." A
**hybrid (GDN+DSA)** loop matches the full-attention loop's quality at fully
linear-time cost (~5.7× decode throughput at 8k); **hybrid (Full+GDN)** *beats* it
by +2.1 pts. They also convert a pretrained Ouro LT into **Ouro-Hybrid-1.4B** with
~1B tokens, matching 1B-class and approaching 4B-class models at linear-time speed.

## Context & motivation

Two scaling axes: parameters (classic) and **depth-via-weight-shared recurrence**
(looped / Universal Transformers — Ouro). Looping is parameter-efficient
reasoning: repeated computation = effective depth at fixed unique params. But each
loop re-applies quadratic full attention over the whole sequence, so training
attention-FLOPs and inference KV-cache **grow with sequence length and compound
per loop** (Fig. 2). At long context the quadratic term dominates and adding loops
becomes impractical. LT2 removes that bottleneck by replacing the mixer.

## Method

### Architecture
A standard LT loops shared blocks: $\mathbf{h}^{(\tau)} = (\mathcal{F}_N\circ\cdots\circ\mathcal{F}_1)(\mathbf{h}^{(\tau-1)})$
for $\tau=1..T$, giving effective depth $T\cdot N$ at a $T\times$ parameter
reduction. LT2 just replaces the MHA sub-layer with any subquadratic
`LinearMixer` (Table 1 lists the supported family: LA, RetNet, Mamba2, GLA,
HGRN2, DeltaNet, **GDN**, **KDA**; sparse: Window, **NSA**, **DSA**).

Two levels of residual: the usual per-block identity residual, **plus a
zero-initialized, per-channel learned per-loop residual** $\mathbf{h}^{(\tau)} = \tilde{\mathbf{h}}^{(\tau)} + g\odot\mathbf{h}^{(\tau-1)}$.
An **SDPA output gate** mitigates attention-sink accumulation under looping.

### Core theoretical claims (§2.2, App. B)
- **Loop × DPLR linear attention → rank-$T$ memory update.** A DPLR mixer (GDN/KDA/RWKV7) has state transition $\mathbf{A}_t = \text{Diag}(\alpha_t)(\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^\top)$, a rank-1 perturbation. Looped $T$ times, the effective operator is $\mathbf{A}_t^{\text{eff}} = \prod_{\tau=1}^T \mathbf{A}_t^{(\tau)}$. Following DeltaProduct, if the loop-specific keys are **orthogonal** across iterations, the rank-1 erasure becomes a **rank-$T$** memory-erasure subspace — strictly more expressive (a single DPLR block can't solve the word problem for ≥3 elements; looping lifts this).
- **Loop × sparse attention → receptive-field expansion.** A window-$w$ block sees the last $w$ tokens per loop; looping chains windows so after $T$ loops $|\mathcal{I}_t^{(T)}| = O(Tw)$ — reaching as far as $T$ stacked window-attention layers but with $T\times$ fewer params.

### Hybrid LT2 (§2.3)
Looping opens a **second hybridization axis**: vary the mixer not only along
**depth** (interleave full-attention layers among linear ones inside the shared
block) but also across **loop iterations** (e.g. a full-attention loop first, then
shrinking-window loops 256→128). Two configs studied: **(GDN+DSA)** = linear +
sparse, no full attention, max efficiency; **(Full+GDN)** = small fraction of full
attention, max quality.

### Training procedure
Pre-train on FineWeb-Edu at **0.6B and 1.3B params, 100B-token budget, $T=4$
loops**; hybrid ratio **1:4** for both Full:Linear and GDN:DSA. Ouro-Hybrid-1.4B:
distill/convert pretrained Ouro (full-attention LT) with **~1B tokens** continued
training.

## Experimental setup

- **Datasets:** FineWeb-Edu pretraining; zero-shot suite (ARC-E/C, HellaSwag, PIQA, WinoGrande, OBQA, SciQ, BoolQ); synthetic recall + state-tracking; long-context retrieval; efficiency (decode throughput, KV memory).
- **Reference:** standard full-attention Looped Transformer; plain Transformer; industry 1B–4B open models (for the converted model).

## Key results

- **Subquadratic ≈ full-attention loop:** looped GDN / KDA / DSA all land within ~1 avg point of the full-attention LT at both scales; at 1.3B, **looped GDN surpasses** the full-attention LT while staying linear-time. (0.6B: Looped Transformer 56.42 avg / 11.92 PPL; Looped GDN 55.74 / 12.06; Looped DSA 55.67 / 12.08.)
- **Hybrid (GDN+DSA)** — *no full attention* — matches the full-attention loop (**9.72 vs 9.87 PPL** at 1.3B; ~59.3% avg), with **~5.7× decode throughput at 8k** (125 vs 22 tok/s, bs 8) and **2.9× at 32k**.
- **Hybrid (Full+GDN)** beats the full-attention LT by **+2.1 pts (61.4% vs 59.3%)** with ~5× decode throughput, winning across LM, recall, state-tracking, and efficiency.
- **Ouro-Hybrid-1.4B** (converted, ~1B tokens): retains the full-attention teacher's quality, inherits linear-time efficiency, matches/exceeds 1B-class baselines and approaches 3B–4B models. Practitioners need not retrain from scratch.
- Gating matters: in the looped setting both gating and DPLR help, with **gating appearing more important than the DPLR update**; pure DeltaNet/RetNet were unstable (RetNet diverged).

## Ablations

- Where to hybridize, mixer arrangement along depth, and mixer ratio (§3.3).
- SDPA output gate vs attention-sink accumulation under looping (§3.4).
- Synthetic recall / state-tracking, long-context efficiency, training stability (§3.5/3.2/3.6).

## Limitations

[analyst] The rank-$T$ expressivity gain is conditional on loop-specific keys
being (near-)orthogonal — an idealized regime; how well trained models realize it
isn't fully pinned down. Pure linear/DeltaNet variants are unstable, so the wins
lean on gating (GDN/KDA) and on keeping *some* full or sparse attention for recall.
Scale tops out at 1.3B; $T=4$ fixed (adaptive computation time deferred to App. A).

## Why it matters [analyst's view]

This is the efficiency-side complement to the looped-transformer thread
([[topics/looped-transformers]], [[topics/recursive-reasoning]]). The slogan
"looping turns compute into context" is the conceptual payload: it reframes the
quadratic-attention tax on looping as unnecessary, because the *loop itself*
manufactures the long-range mixing that full attention provided — rank-$T$ memory
for linear, $O(Tw)$ receptive field for sparse. That makes looped recurrence a
genuinely *scalable* architecture rather than a small-scale curiosity, and the
Ouro conversion shows it's adoptable without from-scratch retraining. Pairs
naturally with [[papers/chen-2026-training-free-looped]] (looping pretrained
models with no training) and [[papers/lee-2026-looped-diffusion-lm]] (looping in
the diffusion/MDM setting).

## Open questions / things to verify

- Does the rank-$T$ benefit hold empirically (key orthogonality) at scale, or is it best-case theory?
- How far past 1.3B / $T=4$ does the "subquadratic matches full attention" result extend?
- Adaptive loop count (ACT) — how much further does per-token compute allocation help?
- Long-context retrieval: does $O(Tw)$ receptive field actually match true long-range recall, or only approximate it?

## Connections

- Builds on: Ouro / Universal Transformers (looped baseline), GDN/KDA/DSA mixers, DeltaProduct expressivity
- Related: [[papers/chen-2026-training-free-looped]], [[papers/lee-2026-looped-diffusion-lm]], [[papers/wang-2025-hierarchical-reasoning-model]], [[blogs/bansal-kv-cache]] (KV-cache cost — the bottleneck LT2 removes)
- Topic MOCs: [[topics/looped-transformers]], [[topics/recursive-reasoning]]
- Author indices: [[authors/chunyuan-deng]], [[authors/yizhe-zhang]], [[authors/hanjie-chen]]

## Selected quotes

> "Looping therefore turns compute into context: once $T$ is moderately large, a small fixed window already covers long sequences." — §2.2

> "LT2-hybrid (Full + GDN) … improves average zero-shot performance by +2.1 points over the standard looped transformer (61.4% vs. 59.3%) while still achieving ∼5× higher decode throughput." — §1
