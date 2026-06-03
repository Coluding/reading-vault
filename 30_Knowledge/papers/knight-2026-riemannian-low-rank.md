---
type: paper
title: "Riemannian Gradient Descent for Low-Rank Architectures"
authors: ["Nicholas Knight"]
year: 2026
venue: "arXiv:2606.02328"
url: https://arxiv.org/abs/2606.02328
rw_id: 01kt7nbt144a127qda2zcp9cps
topics: [optimization, low-rank]
priority: medium
read_state: skimmed
relevance: ""
added: 2026-06-03
last_updated: 2026-06-03
---

# Riemannian Gradient Descent for Low-Rank Architectures

## TL;DR

A careful, honest study of **Riemannian optimization for rank-factored weight
matrices** ($W = AB^\top$), treating the factorization as an implementation
detail and doing gradient descent directly on the rank-$r$ matrix manifold so rank
is enforced automatically. Knight (NVIDIA, single author) maps out **ten points in
the design space** — 2 geometries for rank-$r$ matrices, 3 for rank-$r$ partial
isometries, plus weight-sharing block variants of all five — and applies them to
the QKVO attention parameters of small language models. The headline finding is a
**negative result, honestly reported**: after tuning learning rates, the
Riemannian methods are *competitive with* but **do not conclusively outperform an
AdamW baseline**, while costing substantially more per iteration. The author
remains "optimistic that there is merit … at larger scales."

## Context & motivation

Almost every weight is a matrix; one exploitable structure is **rank**. The
standard trick $W = AB^\top$ cuts $mn$ params to $mr+nr$ (16× fewer for MHA with
outer dim 4096, inner 128). Conventional training updates $A$ and $B$
independently via Euclidean gradients, ignoring $W$. Knight inverts this: the
goal is to learn a rank-constrained $W$ on the smooth submanifold $\mathbb{R}^{m\times n}_r$,
and $AB^\top$ is just a parameterization. The heuristic hope: by accounting for
parameter relationships (e.g. the gauge freedom $AB^\top = AS\cdot S^{-1}B^\top$
for invertible $S$), the optimizer avoids "meaningless search directions" and
converges faster or better. The whole effort is framed as a **rediscovery**
grounded in the Absil–Mahony–Sepulchre and Boumal textbooks.

## Method

### Problem formulation
Minimize objective $f$ assumed to depend **only on $W = AB^\top$**, not on $A,B$
separately (a sufficient condition for the gradient $G_W$ to exist; the
non-uniqueness then turns out innocuous — same Riemannian gradient regardless).

### Schematic Riemannian gradient descent (§2.1)
A single "embedded" template, modifying *only the optimizer*:
1. Get ambient Euclidean gradient $G_W$ (recovered from $G_A, G_B$ without forming $W$).
2. Derive the Riemannian gradient $G_W^R$ (a submanifold tangent vector).
3. **Momentum** (exponential smoothing) $M \leftarrow \nu M + (1-\nu)G_W^R$, then **normalize** to a trust region $T = -\eta\cdot M/\max(\|M\|, c)$.
4. **Exponential map** — move along the geodesic from $W$ to $W'$ (approximated by a **retraction**).
5. **Parallel transport** $M$ along the geodesic (approximated by a **vector transport**).

### The ten geometries
- **Rank-$r$ matrices:** *embedded geometry* (Frobenius metric, metric-projection retraction via truncated SVD of $W+\Xi$ — credited to Vandereycken) and *quotient geometry* (metric defined "upstairs" on $\mathbb{R}^{m\times r}\times\mathbb{R}^{n\times r}$, additive retraction $(A+\Xi_A, B+\Xi_B)$, pushed down through the submersion $(A,B)\mapsto AB^\top$).
- **Rank-$r$ partial isometries:** three geometries handling orthogonality constraints (relevant when one factor is constrained orthonormal).
- **Weight sharing / block variants:** all five adapted to share factors across block-rows/columns (e.g. GQA, where K/V are shared across query heads).

All methods keep **per-iteration complexity $O(mr^2 + nr^2)$** and never form
$m\times n$ matrices — at the cost of "linear-algebraic prestidigitation" and
several lurking numerical hazards (notably "running off the end of a geodesic").

### Training procedure
Cross-entropy LM objective on **FineWeb** ('sample-10BT'), GPT-2 tokenizer,
sequence length 512, batch size 8, **10,000 steps, 3 seeds**. Baseline **AdamW**
(PyTorch defaults), LR grid-searched. Then AdamW on the **QKVO** params is swapped
for each low-rank optimizer and its LR swept; AdamW retained on all other params.
Two architecture cases: **MHA** and **GQA**. No LR annealing; ad-hoc 10-step warmup.

## Key results

- **MHA / GQA validation loss ≈ 6.37–6.38 nats**; across the 10 design points, **the choice of geometry, momentum, and normalization made little difference** (inter-seed std 0.001–0.013 nats; geometry differences within seed variance).
- Improvements over AdamW were "**borderline explainable by seed variance**," somewhat stronger for GQA than MHA.
- AdamW converges **slightly faster** on MHA (embedded geometry the best of the Riemannian set); on GQA the methods track AdamW closely and occasionally edge it out — until a shared step-6660 loss spike.
- **Loss-spike forensics:** two of three MHA AdamW runs spiked ~50× near step 6660 (recovering); the spike coincided with a window of unusually structured documents (lists, Wikipedia tables, a Grateful Dead setlist, a recipe). Disabling weight decay did not remove the spikes.

## Limitations

Author's own, unusually candid: (1) **model and token horizon are tiny** by modern
standards — behavior at scale unknown; (2) **no learning-rate annealing** (known to
matter), plus ad-hoc warmup/batch choices; (3) **AdamW left at defaults**, not
tuned, and no comparison to other optimizers (e.g. Muon). (4) Several retraction /
vector-transport accuracy questions deferred. [analyst] Higher algorithmic
complexity than AdamW is the decisive practical strike given equal data efficiency.

## Why it matters [analyst's view]

A valuable *negative* result done right: it rules out (at small scale) the
intuition that respecting rank-manifold geometry buys optimization gains, and it
does so with enough rigor and self-criticism to be trustworthy. The most
generative thread is the connection it draws in §7: AdamW ≈ steepest descent under
the $\ell_\infty$ norm, and **Muon ≈ steepest descent for matrices under the
spectral (Schatten-$\infty$) norm** — situating Riemannian metrics as yet another
choice of "geometry of the step," alongside the norm-based view that's driving
modern optimizer research. For the vault this is the first dedicated
[[topics/optimization]] / [[topics/low-rank]] entry; if a Muon or
modular-norm paper lands, these become a real cluster. Connects loosely to the
low-rank-adaptation lineage (LoRA-style $AB^\top$) but is about *training geometry*,
not adapter efficiency.

## Open questions / things to verify

- Does any geometry pull ahead **at larger scale** with proper LR annealing? (The author's standing bet.)
- Head-to-head with **Muon** (the spectral-norm steepest-descent method it explicitly invokes) on the same QKVO params.
- Can the methods be extended to architectures that are only **"close to" low-rank** — RoPE, QK-norm (flagged as future work)?
- Is the $O(mr^2+nr^2)$ overhead ever worth it vs the conventional $O(mr+nr)$ update running more steps?

## Connections

- Builds on: Absil–Mahony–Sepulchre & Boumal (Riemannian optimization textbooks), Vandereycken (fixed-rank embedded geometry)
- Related: optimizer-geometry / steepest-descent-under-norms line (AdamW, Muon)
- Topic MOCs: [[topics/optimization]], [[topics/low-rank]]
- Author index: [[authors/nicholas-knight]]

## Selected quotes

> "After tuning learning rates, our methods do not conclusively outperform an AdamW baseline." — Abstract

> "AdamW, the reference optimizer we failed to beat experimentally, is closely related to steepest descent under the $\ell_\infty$ norm. And Muon … can be viewed as steepest descent, for matrix parameters, under the spectral (Schatten-$\infty$) norm." — §7
