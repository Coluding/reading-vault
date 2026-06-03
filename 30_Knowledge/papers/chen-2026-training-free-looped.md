---
type: paper
title: "Training-Free Looped Transformers"
authors: ["Lizhang Chen", "Jonathan Li", "Chen Liang", "Ni Lao", "Qiang Liu"]
year: 2026
venue: "arXiv:2605.23872"
url: https://arxiv.org/abs/2605.23872
rw_id: 01kt7nfs5380dtvjfzqznqrf2p
topics: [looped-transformers, recursive-reasoning, inference-time-scaling]
priority: high
read_state: skimmed
relevance: ""
added: 2026-06-03
last_updated: 2026-06-03
---

# Training-Free Looped Transformers

## TL;DR

A lightweight **inference-time wrapper** that loops a contiguous mid-stack block
of a **frozen** checkpoint — no fine-tuning, no continued training, no new
parameters, no architecture change. Naive block re-application *degrades*
performance; the fix comes from a numerical-analysis lens: a **pre-norm
transformer layer is exactly one forward-Euler step ($h{=}1$) on a per-block
residual ODE**, so re-applying the block is a **finer integration** of the ODE the
network already approximates ($K$ sub-steps of size $1/K$ better hit the same
$t{=}1$ endpoint). Using $K$-stage **Runge–Kutta** on the middle 4 layers
(block-mode for dense, layer-mode for MoE) with **zero per-task tuning**, it
improves modern LLMs across 7 families / 45 (model, benchmark) cells — e.g.
Qwen3-4B-Instruct **+2.64 pp on MMLU-Pro**, +2.01 on GPQA-Main.

## Context & motivation

Looped / Universal Transformers and DEQ models bake recurrence into the
*trained* weights — layers are tied across iterations during optimization, so
looping is inseparable from training. That means **you can't loop almost any
publicly released checkpoint** (Qwen3, Llama-3.2, Moonlight, DeepSeek-V2-Lite are
all standard non-looped stacks). Core question: *can we loop a frozen off-the-shelf
model at inference, with no updates of any kind?*

Converging evidence says maybe: whole mid-layer blocks can be **deleted** with
little loss; mid-layers can be **skipped/swapped/repeated** without catastrophe;
intermediate-layer logits already encode much of the final prediction. The usual
reading is **compressibility** (mid-layers are redundant → removable). The paper's
flip: the same redundancy that makes a layer safe to *delete* makes it safe to
**re-apply**.

## Method

### Loop wrapper
Split a pretrained $f = L_{N-1}\circ\cdots\circ L_0$ into **pre-loop**,
a **loop window** $g = L_b\circ\cdots\circ L_a$, and **post-loop**:

$$\hat{f}(x) = (L_{N-1}\circ\cdots\circ L_{b+1})\circ g^{(K)}\circ(L_{a-1}\circ\cdots\circ L_0)(x)$$

Cost is $(b-a+1)(K-1)$ extra forward passes through the window. No weights touched.

### Block-mode vs layer-mode
- **Block-mode:** $g^{(K)}_\text{blk} = (L_b\circ\cdots\circ L_a)^K$ — iterate the whole window as a unit.
- **Layer-mode:** $g^{(K)}_\text{lyr} = L_b^K\circ\cdots\circ L_a^K$ — iterate each layer $K$ times before moving on.

Dense models: the two are similar. **MoE models: block-mode is unstable** — each
iteration perturbs the hidden state, so every MoE gate **re-routes to different
experts**, and accumulated routing noise overwhelms the refinement. Layer-mode
computes routing **once** and applies the same expert mixture $K$ times → the
correct default for MoE backbones.

### Loop strategies = ODE integrators (§2.3)
A pre-norm layer $L(x) = x + \text{Attn}(\text{LN}_1(x)) + \text{MLP}(\dots)$ is
input + residual = one forward-Euler step. Looping is then just integrating that
residual ODE more finely. They instantiate **7 loop strategies** as known
integration / fixed-point methods: forward Euler, damped sub-steps, **Anderson
acceleration**, heavy-ball, **Aitken acceleration**, and **Runge–Kutta**-style
updates (Algorithms 1–2). A toy 2-D regression (Fig. 2) shows naive $K$-looping
drifts endpoints *outward* into high-loss regions as $K$ grows, while $K$-stage RK
integration stays in the trained low-loss valley.

### Out-of-the-box recipe
$K$-stage Runge–Kutta on the **middle 4 layers**; block-mode for dense, layer-mode
for MoE; **no per-cell hyperparameter tuning**.

## Experimental setup

- **Models (7 families):** Qwen3 (0.6B/1.7B/4B base & instruct, 30B-A3B MoE), Qwen1.5-MoE-A2.7B, Llama-3.2 (1B/3B), Moonlight (16B-A3B), DeepSeek-V2-Lite — dense, sparse MoE, and MLA+MoE.
- **Coverage:** 45 (model, benchmark) cells; knowledge-heavy multiple-choice and reasoning benchmarks (MMLU-Pro, GPQA-Main, ARC-Challenge, CommonsenseQA, OpenBookQA, …).
- **Compute:** ~20,000 NVIDIA H100 80GB GPU-hours.

## Key results

- **Qwen3-4B-Instruct:** +2.64 pp MMLU-Pro, +2.01 pp GPQA-Main.
- **Qwen3-30B-A3B-Instruct (MoE):** +1.14 pp CommonsenseQA.
- **Moonlight-16B-A3B-Instruct (MoE):** +1.20 pp OpenBookQA.
- **Qwen1.5-MoE-A2.7B-Chat:** +2.30 pp ARC-Challenge.
- Gains are strongest on **knowledge-heavy multiple-choice** tasks, achieved with **no parameter updates, no extra supervision, no benchmark-specific tuning** — i.e. repeated application of existing blocks **exposes latent inference-time computation**.

## Ablations

- **Block- vs layer-mode** across dense and MoE (layer-mode necessary for MoE; §2.2).
- **Integrator comparison:** forward Euler vs Anderson / heavy-ball / Aitken / Runge–Kutta — RK is the recommended default.
- Loop window placement (middle layers) and loop count $K$.

## Limitations

[analyst] Gains are real but **modest** (~1–2.6 pp) and concentrated on
multiple-choice knowledge benchmarks; generative/open-ended impact is less
clear. There's a real **inference-cost tax** — $(b-a+1)(K-1)$ extra passes — so
this trades compute for accuracy, competing directly with just sampling more or
using a bigger model. The ODE story is a *motivation*, not a guarantee the frozen
weights actually parameterize a well-behaved flow; naive looping failing is itself
evidence the geometry is fragile.

## Why it matters [analyst's view]

The cleanest "free lunch" framing in the looped-transformer thread
([[topics/looped-transformers]], [[topics/recursive-reasoning]]): you don't need
to *train* looped — you can retrofit recurrence onto any released checkpoint at
test time. The forward-Euler/ODE interpretation is the load-bearing idea and ties
looping to the DEQ / Neural-ODE lineage; it also reframes the layer-pruning
literature (mid-layers as redundant) into "mid-layers as re-applicable." Strong
complement to [[papers/deng-2026-lt2-looped]] (which makes *trained* looping
scalable and efficient) and [[papers/lee-2026-looped-diffusion-lm]] (looping in
diffusion) — together they cover train-from-scratch, convert-pretrained, and
zero-train looping. The MoE routing-thrash finding is a concrete, reusable
gotcha for anyone iterating computation over sparse models.

## Open questions / things to verify

- Does the benefit extend to **generative / long-form reasoning**, or only multiple-choice?
- Compute-matched comparison: is looping better than best-of-$n$ sampling or a slightly larger model at equal FLOPs?
- How robust is the ODE picture — do gains track integration order as the theory predicts across all 7 families?
- Can the loop window / $K$ be chosen adaptively per input rather than fixed at "middle 4 layers"?

## Connections

- Builds on: DEQ / Neural-ODE view of residual nets; layer-pruning & layer-redundancy results
- Related: [[papers/deng-2026-lt2-looped]], [[papers/lee-2026-looped-diffusion-lm]], [[papers/wang-2025-hierarchical-reasoning-model]] (DEQ-style 1-step gradient), [[papers/baek-2026-gram]]
- Topic MOCs: [[topics/looped-transformers]], [[topics/recursive-reasoning]]
- Author indices: [[authors/lizhang-chen]], [[authors/qiang-liu]]

## Selected quotes

> "The same redundancy that makes a mid-layer safe to delete makes it safe to re-apply." — §1

> "Each pre-norm transformer layer is exactly one forward Euler step at $h=1$ on a per-block residual ODE, so re-applying the block at inference is, geometrically, a finer integration of the ODE the network already approximates." — §1
