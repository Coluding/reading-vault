---
type: blog
title: "Learning the integral of a diffusion model"
author: "Sander Dieleman"
url: https://sander.ai/2026/05/06/flow-maps.html
rw_id: 01ktspgqg2drmy8w0rkd9k0j7q
topics: [diffusion-models, flow-matching, distillation, generative-models]
priority: high
read_state: queued
added: 2026-06-10
last_updated: 2026-06-17
---

# Learning the integral of a diffusion model

## TL;DR

Diffusion sampling is the numerical integration of a velocity field: at each
step a denoiser predicts the tangent direction of the (deterministic) path
connecting a noise sample to its data sample, and we accumulate those tangents.
Dieleman's post is a survey + unifying tutorial on **flow maps** — neural
networks $F(\mathbf{x}_s, s, t)$ that learn to predict *any* point on a path
from *any other* point, i.e. the integral of the velocity over a finite time
interval, so generation can be done in one (or few) steps. The central
organizing claim, borrowing the taxonomy of Boffi, Albergo & Vanden-Eijnden, is
that *every* flow-map training method reduces to enforcing one of three
equivalent consistency rules — **compositionality**, **Lagrangian consistency**,
and **Eulerian consistency** — each of which, with the right boundary
conditions, is sufficient on its own to guarantee a valid flow map. The post
shows how each rule yields a squared-residual loss, how `stop-gradient` /
finite-difference / JVP tricks turn these self-referential losses into
tractable static-target regression, and then maps essentially the entire 2024–26
literature (consistency models, MeanFlow, Shortcut models, Align Your Flow,
FreeFlow, TVM, FMM, …) onto that three-way grid. The recurring "no free lunch"
moral: sampling from a flow map is cheap, but training one is harder than
diffusion and almost always bootstraps from a diffusion model's velocity.

## Context

This is a long technical explainer on Sander Dieleman's blog (sander.ai), a
follow-up to his earlier posts on diffusion distillation, the geometry of
guidance, and "perspectives on diffusion." It assumes diffusion familiarity and
comfort with vector calculus. The motivation is that two-ish years after his
distillation post, the literature has exploded with new few-step / one-step
sampling methods under a confusing thicket of names and formalisms ("semigroup
property," "shortcut," "stochastic interpolants," etc.), and he wants to impose
order. He explicitly adopts the **flow map** framing and the three-consistency-
rule **taxonomy of Boffi et al.** (flow map matching + self-distillation papers)
but deliberately translates everything out of the "stochastic interpolant"
language back into a "traditional diffusion" framing he expects more readers to
share. The piece is part conceptual derivation, part annotated bibliography.

## Core argument

**1. Diffusion = integrating a velocity field along non-crossing paths.**
Given a denoiser $\hat{\mathbf{x}}_0 = f(\mathbf{x}_t, t) = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$,
deterministic sampling (DDIM / probability-flow ODE / flow matching) defines a
**bijection** between noise and data: each noise sample maps to exactly one data
sample via a unique path, and paths never cross (a crossing would create
ambiguity in the tangent direction, which is impossible since the denoiser is
*memoryless* and *myopic* — it sees only the current $(\mathbf{x}_t, t)$).
Traversal is "dead reckoning." Using the flow-matching schedule
$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\varepsilon}$, $T=1$,
$\boldsymbol{\varepsilon}\sim\mathcal{N}(0,1)$ (with the diffusion time convention
$t=0$ = data, $t=1$ = noise), the velocity is
$v(\mathbf{x}_t, t) = \mathbb{E}[\boldsymbol{\varepsilon} - \mathbf{x}_0 \mid \mathbf{x}_t]$,
a linear function of $\hat{\mathbf{x}}_0$ and $\mathbf{x}_t$.

**2. A flow map is the integral of that velocity.**

$$F(\mathbf{x}_s, s, t) = \mathbf{x}_s + \int_s^t v(\mathbf{x}_\tau, \tau)\, d\tau$$

Adding the accumulated tangent to the start $\mathbf{x}_s$ lands you at
$\mathbf{x}_t$. Special cases: $F(\mathbf{x}_s, s, 0) = \mathbf{x}_0$ is one-step
sampling (what **consistency models** do, a special case of flow maps);
$F(\mathbf{x}_t, t, t) = \mathbf{x}_t$ (zero-length interval); $t > s$ runs
backwards toward noise. The equivalent **average-velocity / mean-flow**
parameterization is
$V(\mathbf{x}_s, s, t) = \frac{1}{t-s}\int_s^t v\, d\tau$, related by
$F(\mathbf{x}_s, s, t) = \mathbf{x}_s + (t-s)\,V(\mathbf{x}_s, s, t)$, with the
limit $V(\mathbf{x}_t, t, t) = v(\mathbf{x}_t, t)$ — so **a flow map contains a
denoiser inside it**. Flow maps are a strict generalization of diffusion models,
but every known way to build one bootstraps from the local (velocity) view.

**3. Three consistency rules — any one suffices.** A network of three inputs
can fit almost anything; we must explicitly enforce that it *is* a valid flow
map. The post derives three equalities and shows Lagrangian and Eulerian are
each just compositionality with one interval shrunk to infinitesimal:

- **Compositionality** (a.k.a. semigroup / shortcut / progressive):
$F(F(\mathbf{x}_s, s, t), t, u) = F(\mathbf{x}_s, s, u)$. Corollary: a flow map is
its own inverse, $F(F(\mathbf{x}_s, s, t), t, s) = \mathbf{x}_s$.
- **Lagrangian consistency** (move the target $t$, watch the output travel
along the path): $\frac{\partial}{\partial t} F(\mathbf{x}_s, s, t) = v(F(\mathbf{x}_s, s, t), t)$.
The instantaneous change of the output equals the velocity. Derived from
compositionality by letting $u \to t$ and using the fundamental theorem of
calculus.
- **Eulerian consistency** (fix the target $t$, change the source $s$ — output
must not move): $\frac{\partial}{\partial s} F = 0$, which via the multivariate
chain rule (since both $s$ and $\mathbf{x}_s$ depend on $s$, and
$\frac{d\mathbf{x}_s}{ds} = v(\mathbf{x}_s, s)$) becomes
$\frac{\partial}{\partial s} F(\mathbf{x}_s, s, t) + \nabla_{\mathbf{x}_s} F(\mathbf{x}_s, s, t)\, v(\mathbf{x}_s, s) = 0$.
Lagrangian and Eulerian are "the same river" from two reference frames — sitting
in the canoe vs. standing on the bridge.

**4. Equalities → losses.** Move everything to one side, square the residual,
average over time steps and data:
$\mathcal{L}_\text{compositional} = \mathbb{E}[(F(F(\mathbf{x}_s,s,t),t,u) - F(\mathbf{x}_s,s,u))^2]$,
$\mathcal{L}_\text{Lagrangian} = \mathbb{E}[(\frac{\partial}{\partial t}F - v(F,t))^2]$,
$\mathcal{L}_\text{Eulerian} = \mathbb{E}[(\frac{\partial}{\partial s}F + \nabla_{\mathbf{x}_s}F\, v(\mathbf{x}_s,s))^2]$.
With a boundary constraint $F(\mathbf{x}_t,t,t)=\mathbf{x}_t$ and a meaningful
velocity (from a pre-trained teacher → distillation, or learned jointly →
from-scratch), the minimum of any of these is a valid flow map.

**5. The losses are weird and need taming.** They are **self-referential with
moving targets** — unlike ordinary regression against a static ground truth.
Lagrangian/Eulerian contain derivatives *of the function being learned* (risking
higher-order backprop); compositional requires sequential applications of $F$.
The fix, borrowed from self-distillation in representation learning, is the
**stop-gradient** ($\mathrm{sg}[\cdot]$): wrap the target side so it's treated as
a frozen constant. This turns the loss into ordinary regression-to-a-frozen-
target, avoids higher-order derivatives, and can be combined with **EMA teacher
weights** for stability. The catch: the resulting update is no longer a true
gradient of the stated loss — it's a **semigradient**, so theoretical guarantees
lapse (but it works in practice).

## Notable details

- **MeanFlow (MF)** trains a flow map *from scratch* without distillation, via
the **marginal-from-conditional ("marginalisation") trick**. Starting from the
average-velocity form of the Eulerian rule and wrapping the RHS in
`stop-gradient`, you may substitute the marginal velocity $v(\mathbf{x}_s,s)$ by
the *conditional* velocity $\boldsymbol{\varepsilon}-\mathbf{x}_0$ without moving
the MSE minimum. This is only valid because of four features: (i) MSE loss, (ii)
velocity evaluated *at $\mathbf{x}_s$* (so the expectation is conditional on
$\mathbf{x}_s$), (iii) target linear in $v$, (iv) the `stop-gradient` keeps the
*update direction* $\widetilde{G}_\theta = \mathbb{E}[2R\,\nabla_\theta V]$ linear
in $v$. Dieleman stresses the `stop-gradient` does "double duty": it both avoids
higher-order differentiation *and* is what makes marginal-from-conditional
learning work at all. The same trick fails for the Lagrangian rule because there
velocity is evaluated at $\mathbf{x}_t = F(\mathbf{x}_s,s,t)$, the wrong
conditioning variable.
- **Improved MeanFlow (iMF)** swaps $V$ and $v$ to opposite sides, then
substitutes $v(\mathbf{x}_s,s) = V(\mathbf{x}_s,s,s)$, defining a reparameterized
predictor $W$ that you train *as if it were a plain diffusion/flow-matching
model* with the standard
$\mathcal{L}_\text{iMF} = \mathbb{E}[(W(\mathbf{x}_s,s,t) - (\boldsymbol{\varepsilon}-\mathbf{x}_0))^2]$.
The flow map "falls out" purely through the parameterization. Elegantly, the
`stop-gradient` (on the JVP) is here *not necessary for correctness*, only for
efficiency — and iMF has much lower loss variance.
- **JVP / autodiff plumbing.** The Eulerian Jacobian-vector product
$\nabla_{\mathbf{x}_s} V \cdot v$ multiplies a $K\times K$ Jacobian (for $K$-dim
inputs) that you never want to materialize; **forward-mode** differentiation
(e.g. `jax.jvp`) computes it with the tangent vector $[v,1,0]$ alongside the
forward pass. Higher-order derivatives are flagged as costly (FLOPs + memory),
often meaningless (ReLU nets have ~zero curvature a.e.), and unsupported by fast
kernels like **FlashAttention** — motivating the whole `stop-gradient` / finite-
difference apparatus. **Terminal Velocity Matching (TVM)** is the notable
exception that *does* keep the higher-order derivative and ships a **custom
attention kernel** plus a Lipschitz constraint.
- **The taxonomy table** (the post's centerpiece) sorts methods by which rule
they enforce. *Lagrangian* 🐱: LMD, Lagrangian Self-Distillation (LSD),
Align-Your-Flow-LMD, TVM, **FreeFlow** (fully *data-free* distillation — samples
only noise + an auxiliary renoising denoiser; argues the training data may not
even represent what the teacher generates under CFG), Physics-Informed
Distillation (PID). *Eulerian* 🐔: EMD, AYF-EMD, SoFlow, **Flow-Anchored
Consistency Models (FACM)** (extend source range to $[0,2]$ so $s>1$ makes the
model act as a denoiser — an auxiliary "anchor" task), MeanFlow + variants
(AlphaFlow curriculum, **Decoupled MeanFlow** — condition early layers on $s$,
late layers on $t$; Rectified MeanFlow — one reflow stage to straighten paths;
Pixel MeanFlow — operate in pixel space when steps are already few). *Compositional*
🐶: **Shortcut models** (avg velocity over $2h$ = mean of two length-$h$
velocities; bootstrapping by step-doubling à la progressive distillation),
SplitMeanFlow, Flow Map Matching (FMM, uses self-inverse identity) and
Progressive FMM, plus the **consistency-model lineage** (CD/CT, iCT, ECT, sCM,
and **CTM** — the first two-time flow map generalizing $t>0$).
- **Guidance is hard for flow maps.** CFG's power comes from compounding small
linear edits across many sampling steps — exactly what distillation removes. The
fix: apply guidance to the *teacher* during distillation (fixed scale, or
**autoguidance**), or **randomize the guidance scale and feed it to the student
as conditioning** (as in iMF, TVM, GFT) so the student learns the effect
directly. Guidance also conveniently simplifies the distribution the flow map
must model.
- **Tricks of the trade:** initialize flow-map weights from a denoiser (or copy
the teacher); consistency mid-training (CMT) to bridge infinitesimal→finite
steps; predict $\hat{\mathbf{x}}_0$ / a "denoised image field" (Pixel MeanFlow)
since data lives on a low-dim manifold while noise doesn't; oversample the $s=t$
case heavily to keep the implied denoiser grounded; adaptive loss weighting as
$s \to t$.

## Why it matters [analyst's view]

The value here is less any single result than the **map of the territory**: a
clean claim that the sprawling few-step-sampling literature is *one idea* (learn
the integral of the velocity field) enforced via *one of three equivalent
constraints*, with the differences between papers being almost entirely a matter
of (a) which rule, (b) which `stop-gradient` / finite-difference / JVP trick
tames the self-referential derivative, and (c) distillation vs. from-scratch.
That compression is exactly the kind of scaffolding that makes the primary
papers readable. It connects directly to the vault's running 2026 theme that the
*dynamics* a generative model implements — not just architecture/scale — are the
design surface (cf. [[topics/generative-models]] MOC, flux-matching, dual-rate
diffusion). The marginal-from-conditional analysis of MeanFlow is the most
quietly important bit: it shows a `stop-gradient` doing load-bearing
*statistical* work (enabling an unbiased conditional-for-marginal substitution),
not just engineering convenience — a subtlety worth remembering whenever a loss
freezes part of itself. The iMF "it's just a reparameterized diffusion model"
framing is the most promising practical takeaway for actually training one. Open
thread for me: the post repeatedly hedges that path-straightness / curvature
claims are "hotly debated," and several methods (AYF-LMD) work on toys but fail
on real images — so the gap between the clean theory and image-scale practice is
still real.

## Connections

- Topic MOCs: [[topics/generative-models]]; [[topics/diffusion-models]]; [[topics/flow-matching]]; [[topics/distillation]]
- Related notes: [[blogs/shing-diffusionblocks]] (diffusion-as-process reframing); [[papers/yang-2026-replaid-continuous-diffusion]] (continuous diffusion LMs, distillation); [[papers/bartosh-2026-dual-rate-diffusion]] (few-step diffusion efficiency, composes with distillation); [[blogs/flow-based-llms-intro]] (flow-matching framing for LLMs)
- Author index: [[authors/sander-dieleman]]

## Selected quotes

> "While diffusion models describe paths between noise and data by predicting
> the tangent direction at each point along the path, flow maps are instead able
> to predict any point on a path from any other point on that same path."

> "It is fair to say that the stop-gradient operation in MeanFlow is doing double
> duty: it avoids higher-order differentiation (no backprop through derivatives),
> and it enables marginal-from-conditional learning. At a glance, it looks like a
> tweak to make training more efficient, but it is actually crucial for training
> to work at all."

> "As ever in machine learning, there is no free lunch: while sampling using a
> flow map is cheaper than sampling from a diffusion model, training a flow map
> is significantly more involved, and often requires training a diffusion model
> first."
