---
type: blog
title: "Improving Neural Network Training by Decoupling the Magnitude and Direction of Weight Vectors"
author: "Alexander Hägele"
url: https://haeggee.github.io/posts/magnitude-direction-decoupling
rw_id: 01kvq3r6t6a63bdasy8443eyvy
topics: [optimization, weight-normalization, training-dynamics, scaling-laws]
priority: medium
read_state: queued
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

The post introduces **Magnitude-Direction Decoupling (MD)**: constrain each weight matrix's *direction* to a fixed-norm sphere while giving it a separately learned *magnitude* (a gain), so an optimizer step controls the relative weight update directly instead of tangling magnitude and direction together. The motivation is that standard optimizers suffer **magnitude-direction interference** — on a scale-invariant loss the same normalized step changes the direction (and thus the loss) far more at small magnitude than at large, while the magnitude drifts upward even though the loss has no radial gradient. Fixing weights onto a sphere already beats the base optimizer's optimal loss, and adding learnable magnitudes gives a further boost. The headline practical payoffs are **automatic learning-rate transfer across model width and depth**, **warmup-free training**, cleaner LR-decay behavior, and improvements that hold in sparse MoEs across a wide compute range. Work by Hägele, Kosson, Hernández-Cano, and Jaggi (EPFL MLO, 2026).

_Note: the fetched Readwise content for this post consists mainly of the figure captions plus the BibTeX; the running prose was not captured. The mechanism and results below are reconstructed faithfully from those captions and should be re-checked against the full post._

## Context

An EPFL MLO (Martin Jaggi's lab) research write-up, 2026, by **Alexander Hägele** (site `haeggee.github.io`, "PhD Student, MLO @ EPFL"), with co-authors **Atli Kosson**, **Alejandro Hernández-Cano**, and **Martin Jaggi** (confirmed via the post's BibTeX: `hagele2026improving`). It sits in the line of work on scale-invariance, weight normalization, and learning-rate transfer / µ-parameterization-style scaling, and it is base-optimizer-agnostic — experiments use both **AdamW** and **Muon** as the underlying optimizer. It reads as the companion blog to a paper of the same title.

## Core argument

**The problem — magnitude-direction interference (Figs. 1–2).** Consider a toy *scale-invariant* loss where only the *direction* of the weight vector affects the loss (the magnitude is irrelevant — there is no radial gradient). Under normalized gradient descent, the *same* step:
- changes the **direction** — and hence the loss — **far more at small magnitude than at large magnitude** (a fixed-size step subtends a larger angle on a small-radius sphere), and
- still **increases the magnitude** over training even though the loss has no radial component.

So in a standard optimizer the effective "learning rate on the direction" is silently coupled to each weight's current magnitude, and magnitudes drift, making the true update scale a moving, uncontrolled quantity.

**The fix — decouple magnitude from direction.** Project the weight matrix's direction onto a **fixed-norm sphere** (normalize it) and reintroduce scale through a **separately learned magnitude / gain**. Once the direction lives on a sphere of fixed radius, an optimizer step maps directly and predictably to a **relative weight update**, decoupled from the magnitude. Fig. 1 (left) shows the ablation cleanly: fixing weights onto a sphere already **improves upon the base optimizer's optimal loss**, and **introducing learnable magnitudes (the "MD" method) gives yet another boost** — independently of whether the base optimizer is AdamW or Muon.

**Design axes (Figs. 3–4.2).** The method has two knobs:
- **Normalization axis** — which slice of a matrix is constrained to fixed norm: each **output row**, each **input column**, or the **whole matrix (Frobenius / flat)**. This governs how the direction is pinned (Fig. 4.1).
- **Gain axis and its parameterization** — how the learnable magnitude acts: **scalar**, **per-row**, **per-column**, or **both rows and columns** (Fig. 4.2, left/middle). Crucially, the gain is best updated **not directly but through a smooth `softplus` map**: the reparameterization **avoids the loss spikes** that a direct gain update produces (Fig. 4.2, right).

**Why it pays off — LR transfer (Figs. 5.1–5.2, 6).** Because the sphere constraint makes the **relative weight update follow the learning rate directly**, the optimal matrix LR **transfers automatically across model width, across depth, and across joint width-and-depth scaling** — the optimal matrix LR stays roughly fixed as the model grows (Fig. 5.1). The mechanism (Fig. 5.2): the relative weight update is exactly controlled across width, which keeps the **relative change in each layer's output stable**, and across depth the **per-layer activation RMS stays well-behaved**. Fig. 6 shows the baseline (changing only the matrix LR, no decoupling) for comparison under AdamW and Muon.

**Further consequences.**
- **LR decay shape matters more on the sphere (Fig. 7):** because the relative update follows the schedule directly, the decay shape has a clearer effect than under weight decay; WSD vs. linear decay are compared, and the relative weight update of the attention query projection tracks the decay shape directly.
- **Warmup-free and continual training (Fig. 8):** dropping warmup actually *improves* the loss, and re-warming runs (on a 150M model) stay stable, with the gradient norm confirming stability.
- **Scale (Figs. 1 middle, 9):** in large sparse **MoEs**, the improvement holds across a wide range of compute (a scaling law of loss vs. FLOPs), and the method also behaves well under batch-size scaling.

## Notable details

- Two-part contribution isolated by ablation: **(1) sphere constraint alone** already beats the base optimizer's optimum; **(2) learnable magnitudes on top** add a further gain.
- The **softplus gain reparameterization** is the detail that makes learnable magnitudes usable — direct gain updates cause **loss spikes**; the smooth map removes them.
- **Base-optimizer-agnostic:** demonstrated on both **AdamW** and **Muon**.
- **LR transfer across width AND depth AND both jointly** — not just width (which is the usual µP claim).
- **Warmup can be dropped entirely** and it *helps*, not just breaks even.
- Constraint granularity (row / column / Frobenius) and gain granularity (scalar / per-row / per-column / both) are independent design axes, each swept.
- BibTeX: `@misc{hagele2026improving, ... author={Hägele, Alexander and Kosson, Atli and Hernández-Cano, Alejandro and Jaggi, Martin}, year={2026}}`.

## Why it matters [analyst's view]

This is a clean, mechanistic take on the same problem µP and weight-normalization attack — making optimization behave predictably as models scale — but it reframes the cure as *geometric*: put the direction on a sphere so a step means a fixed *relative* update, then let a small learned gain carry all the scale information. The appeal for future work is threefold: (1) **LR transfer across depth (not just width)** would remove a real cost in scaling experiments, since it lets you tune once on a small model and reuse the LR at scale; (2) **warmup-free stable training** is a concrete simplification with a clear stated mechanism (controlled relative updates ⇒ stable activation RMS across layers), not just an empirical trick; (3) the result that improvements survive into **sparse MoEs across a compute range** is the evidence that matters most for whether this is production-relevant rather than a toy-scale curiosity. Worth pairing with the vault's scaling-laws material and any future weight-normalization / µP notes. Caveat: my summary is reconstructed from figure captions (the prose body wasn't captured in the fetch), so treat the exact framing as provisional and verify against the full post and its paper before citing specific claims.

## Connections

- Topic MOCs: [[topics/scaling-laws]]
- Related papers: _none clearly in the vault yet — a weight-normalization (Salimans & Kingma) or µP scaling note would be the natural neighbor; `_needs note_`_
- Author index: [[authors/alexander-hagele]]

## Selected quotes (optional)

> "Independently of the base optimizer, fixing weights onto a sphere improves upon the optimal loss; introducing learnable magnitudes (our work, MD) gives yet another boost." — Fig. 1

> "the identical step changes the direction — and hence the loss — far more at small magnitude than at large magnitude ... even though the loss has no radial gradient, the step still increases the magnitude." — Fig. 2

> "By controlling the relative weight updates directly through the sphere, the optimal LR automatically transfers." — Fig. 1 (right)
