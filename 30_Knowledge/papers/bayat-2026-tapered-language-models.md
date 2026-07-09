---
type: paper
title: "Tapered Language Models"
authors: ["Reza Bayat", "Ali Behrouz", "Aaron Courville"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2606.23670
rw_id: 01kwd5v4r6m7hmycp4pravfdvt
topics: [language-models, scaling-laws, efficient-training]
priority: medium
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

> **Sourcing note:** the Readwise `html_content` cached under this rw-id was cross-wired to a *different* paper ("The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show"), so this note was written from the paper's own arXiv HTML (`arxiv.org/html/2606.23670`, published 2026-06-22) rather than the Readwise export. Equations are transcribed from that rendering; exact perplexity/accuracy numbers should be double-checked against the PDF before quoting.

## TL;DR

Transformers (and their modern relatives) allocate the same MLP width to every layer. This paper argues that's wasteful: **early layers deserve more capacity and late layers less.** They introduce **Tapered Language Models (TLMs)**, which vary the MLP intermediate dimension $d_{ff}(l)$ *monotonically down* with depth via a **cosine schedule**, holding total parameters and FLOPs fixed. The best profile gives the first layer $1.5\times$ and the last layer $0.5\times$ the baseline width. On a 440M Transformer, cosine tapering cuts validation perplexity from **16.28 → 14.44** at zero added cost, and the same fixed schedule transfers to 760M/1.3B models and to four different architectures (Transformer, Gated Attention, Hope-attention, Titans), improving perplexity and average commonsense accuracy across the board. The mechanism: probing shows late-layer MLP outputs increasingly just *reinforce* the residual stream rather than add new features, so their capacity is better spent early.

## Context & motivation

Uniform per-layer width is an unexamined default inherited from the original Transformer — every block gets the same $d_{ff}$, typically $4\times$ the hidden size. Prior architecture work has mostly tuned *depth vs width* globally or pruned/distilled after training; comparatively little asks how to *distribute a fixed parameter budget across depth* from the start. The paper's premise is that layers do qualitatively different work: early layers build features, late layers refine and route. If late layers are doing less genuinely new computation, giving them full MLP width is redundant, and that capacity would pay off more in early layers. The contribution is a simple, training-time, budget-neutral reallocation rule plus evidence it generalizes across scales and architecture families.

## Method

### Problem formulation
Fix a total MLP-parameter budget. Choose per-layer intermediate widths $d_{ff}(l)$ for $l=0,\dots,L-1$ that (a) sum to the same total as a uniform baseline and (b) improve loss. Objective: minimize perplexity subject to the budget constraint — a *reallocation*, not a *scaling*, problem.

### Core idea
Replace uniform $d_{ff}$ with a **monotonically decreasing** schedule across depth: more capacity up front, less at the top, keeping the average (hence total params and FLOPs) unchanged.

### Architecture / algorithm
**Budget-preservation constraint** — the mean width equals the baseline:
$$\frac{1}{L}\sum_{l=0}^{L-1} d_{ff}(l) = d_{ff}^{\text{baseline}}.$$

**Cosine schedule (the winning profile):**
$$d_{ff}(l) = d_{end} + \frac{d_{start}-d_{end}}{2}\left(1 + \cos\frac{\pi l}{L-1}\right),$$
where $d_{start}$ is the first-layer width and $d_{end}$ the last-layer width; at $l=0$ the cosine is $1$ giving $d_{start}$, at $l=L-1$ it is $-1$ giving $d_{end}$. The optimal ratio found is $d_{start}/d_{end} = 1.5/0.5$ (earliest layers $1.5\times$ baseline, latest $0.5\times$). Per-layer widths are rounded to the nearest multiple of 16 for hardware efficiency; first/last layers are pinned exactly to $d_{start},d_{end}$.

**What is tapered:** *only* the MLP intermediate dimension. Attention head count, key/value dimension, and recurrent state size are held constant (left to future work). Alternative schedules tested: **linear** (constant-rate decay) and **sigmoid** (transition concentrated mid-stack).

### Derivations / why it works
No formal derivation — the justification is empirical/mechanistic. The authors measure, per layer, the cosine similarity between an MLP block's output and the residual stream it writes into:
$$\rho_l^{\text{MLP}} = \cos\big(\mathcal{F}_l(z_l),\, h_l\big),$$
where $\mathcal{F}_l(z_l)$ is the MLP output at layer $l$ and $h_l$ the residual. Finding: **$\rho_l^{\text{MLP}}$ rises with depth** — later MLPs produce outputs increasingly *aligned* with what's already in the residual, i.e. they reinforce existing content rather than compute orthogonal new features (Pearson $r$ from 0.49 to 0.71 across GPT-2-family models, a strong positive depth association). Interpretation: late-layer MLP capacity at uniform width is underutilized/redundant, so tapering redirects it forward where outputs are still orthogonal and each added dimension does real work.

### Training procedure
- **Architectures:** standard Transformer (softmax attention), Gated Attention (output-gated attention), Hope-attention (nested-learning / self-modifying memory), Titans (attention + neural long-term memory).
- **Scales / tokens:** 440M (30B tokens), 760M (50B tokens), 1.3B (100B tokens).
- **Tokenizer/seq:** Llama-3 tokenizer (32K vocab), 4K-token sequences.
- **Optimizer:** AdamW, cosine LR annealing, peak LR $4\times10^{-4}$, weight decay 0.1, global batch 0.5M tokens.

### Inference / sampling
Standard autoregressive LM decoding; tapering changes only per-layer MLP widths, not the inference procedure.

## Experimental setup

- **In-distribution:** held-out validation split of the training corpus (perplexity).
- **OOD perplexity:** WikiText-2, LAMBADA.
- **Downstream (8 commonsense benchmarks):** PIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, SIQA, BoolQ, LAMBADA-accuracy.
- **Baseline:** uniform-width model at the *same* parameter/FLOP budget.

## Key results

*Summarized — verify exact figures against the PDF.*
- **Schedule sweep, 440M Transformer:** uniform 16.28 ppl → **cosine (1.5→0.5) 14.44** (−1.84); linear (1.5→0.5) 15.96 (−0.32); sigmoid (1.5→0.5) 16.12 (−0.16). Cosine is decisively best.
- **Direction matters:** in a 3-group (early/mid/late) 440M experiment, wider-early gave 15.96 ppl vs 16.28 uniform, while **wider-late hurt badly — 17.29** (+1.0 over baseline). Front-loading helps; back-loading harms.
- **Transfer to larger models & architectures** (fixed cosine 1.5/0.5): WikiText perplexity improved in 7/8 comparisons, LAMBADA in 8/8. Average commonsense accuracy gains:
  - 760M — Transformer 52.25→52.84 (+0.59), Gated Attn 52.61→52.88 (+0.27), Hope 53.69→54.05 (+0.36), Titans 52.30→53.29 (+0.99)
  - 1.3B — Transformer 56.05→56.38 (+0.33), Gated Attn 56.51→56.80 (+0.29), Hope 56.95→57.05 (+0.10), Titans 56.73→57.08 (+0.35)

## Ablations

- **Width-ratio × schedule grid:** five ratios (1.25→0.75 … 1.75→0.25) × three schedules = 15 budget-fixed configs. Perplexity traces a **clean U-shape** with the optimum at cosine 1.5→0.5 — both too little and too much tapering are worse, and cosine dominates linear/sigmoid at every ratio.
- The residual-alignment probing ($\rho_l^{\text{MLP}}$ rising with depth) is the mechanistic ablation motivating *why* the U-shape optimum sits where it does.

## Limitations

- **Schedule chosen on 440M Transformer only,** then transferred unchanged to larger models and other architectures. The authors explicitly note this "does not establish that it is optimal for every model" — the best tapering profile likely depends on depth, hidden dimension, and the MLP's share of parameters.
- **Only MLP width is tapered.** Attention heads, KV dimension, and recurrent state size are untouched — plausible additional axes left unexplored.
- [analyst's view] Gains are consistent but modest at the larger scales (+0.1 to +0.35 accuracy at 1.3B); the perplexity story is stronger than the downstream story, and the headline −1.84 ppl is a small-model result.

## Why it matters [analyst's view]

This is a "free lunch if it holds" result: a one-line change to how you set per-layer MLP width, no extra params or FLOPs, that improves loss and transfers across architecture families including non-attention memory models (Titans, Hope). If it replicates at frontier scale it's the kind of default that quietly gets absorbed into everyone's config. Conceptually it dovetails with interpretability findings that late layers do more "refinement/routing" than feature construction — the residual-alignment probe is a nice, cheap diagnostic in its own right. The main open risk is scale: budget-reallocation tricks often shrink as models grow, and the paper's strongest numbers are at 440M.

## Open questions / things to verify

- Exact perplexity/accuracy figures (16.28→14.44, the 8 downstream deltas) — transcribed from the arXiv HTML via an extraction pass; confirm against tables in the PDF.
- Does the −1.84 ppl gain hold, shrink, or grow past 1.3B? Reallocation benefits frequently attenuate with scale.
- Is 1.5/0.5 really transfer-optimal, or would a depth-dependent ratio (the authors' own hypothesized dependence) beat the fixed schedule at 1.3B?
- What happens when you *also* taper attention heads / KV dim / recurrent state — additive gains or interference?
- Interaction with the cosine *LR* schedule (both use cosine — unrelated, but worth confirming no confound in the width sweep).

## Connections

- Topic MOCs: [[topics/language-models]], [[topics/scaling-laws]], [[topics/efficient-training]]
- Author indices: [[authors/reza-bayat]], [[authors/ali-behrouz]], [[authors/aaron-courville]]
- Related: architecture-efficiency / parameter-allocation line of work; the Titans / nested-learning ("Hope") memory architectures it evaluates on.

## Selected quotes (optional)

_Verbatim quotes not preserved — note written from a summarizing extraction of the arXiv HTML rather than the raw text; re-open the PDF for exact quotable passages._
