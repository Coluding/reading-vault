---
type: blog
title: "Review of Andy Clark, Surfing Uncertainty: Prediction, Action, and the Embodied Brain"
author: "Andrew Buskell"
url: https://www.thebsps.org/reviewofbooks/andy-clark-surfing-uncertainty-prediction-action-and-the-embodied-brain/
rw_id: 01kt9h5887vnz8txdtcmna3qxg
topics: [predictive-processing]
priority: medium
read_state: deep
added: 2026-06-04
last_updated: 2026-06-04
---

# Review: Surfing Uncertainty (Andrew Buskell, BSPS Review of Books)

## TL;DR

A scholarly, sympathetic-but-critical review of Clark's book for the *British
Society for the Philosophy of Science*. Buskell frames PP as a **"mid-level
organizational sketch"** — an abstract mechanistic characterization of the nervous
system as a hierarchical generative model that predicts the distal causes of
sensory stimulation — and reconstructs the book's **cumulative inference-to-the-
best-explanation** argument. His sharpest contribution is flagging that Clark gives
*three different, possibly incompatible* accounts of what it means to "move up" the
representational hierarchy.

## Context

A book review (2017) in an academic philosophy-of-science venue. Where Scott
Alexander supplies the accessible intuition and Clark the manifesto, Buskell
supplies the critical-philosophy lens — assessing PP *as a theory*, its
argumentative structure, and where it over-promises.

## Core argument

PP inverts the textbook "detection → enrichment → rich internal representation"
picture. Instead the brain realizes a **generative model** producing hypotheses
about the spatiotemporal structure of sensory input, tuned by Bayesian inference,
with **precision-weighting** modulating top-down priors against bottom-up stimuli
(lineage: Friston, Hinton, Knill, Pouget). Crucially, on Clark's account **what we
experience are the hypotheses produced by the generative model** — but these should
*not* be read as representations that "mirror" the world; ascending the hierarchy
means representing probability distributions over increasingly temporally-extended
trajectories, "a radical departure from … content that corresponds to states of
affairs."

Buskell summarizes the book's structure: Part 1 builds the PP machinery; Parts 2–3
apply it as an extended inference-to-best-explanation across episodic
memory/imagination/dreaming (the "cognitive package deal"), action and planning,
mirror neurons and proprioception, the empiricism-vs-nativism debate, and
neuropathology (schizophrenia, autism, depersonalization) — then situate PP within
cultural evolution, enactivism, and embodied cognition ("philosophy of nature").

## Notable details

- **The hierarchy ambiguity (his main critique):** Clark characterizes "moving up" the hierarchy three ways that aren't obviously the same — (a) increasing **spatiotemporal scope**, (b) increasingly sketchy high-level **gists/context** topping out in a background "mindset," and (c) **hyperpriors** encoding "almost 'Kantian'" abstract features of the world. Spatiotemporal scope ≠ gist ≠ invariant feature; their compatibility is asserted, not shown.
- Invokes **Holton's "themata"** — early, unfalsifiable framework commitments — to characterize the bottom-up-vs-top-down dispute: much of the disagreement is thematic, not empirical, and committed "bottom-up" theorists can always craft saving hypotheses (e.g. efference-copy accounts of action).
- Honest about over-reach: the breadth means many literatures (mirror neurons, context, imitation; folk psychology, disjunctivism, realism) are "treated far too hastily" — best read as **promissory notes**, not knock-down arguments. Most-used phrase in the book: "The upshot is…".
- Verdict: an **"agenda-setter"** in the lineage of Clark's *Microcognition* and *Being There* — important not because it settles things but because it sets the stage for debates to come.

## Why it matters [analyst's view]

The valuable counterweight in this 3-note cluster: it keeps PP honest. The
hierarchy ambiguity Buskell isolates is directly relevant to anyone importing PP
intuitions into ML — "deeper layer = bigger spatiotemporal scope" vs "deeper layer
= more abstract gist" vs "deeper layer = stronger prior" are *different
architectural commitments*, and conflating them is exactly the kind of slippage
that makes "the brain does predictive coding, so our net should too" arguments
mushy. The themata framing is a useful reminder that bottom-up vs top-down (or,
in ML terms, discriminative-feedforward vs generative-predictive) is partly a
prior about how to model, not a settled empirical fact.

## Connections

- Topic MOCs: [[topics/predictive-processing]]
- Companion notes (same book): [[blogs/clark-surfing-uncertainty-precis]], [[blogs/alexander-surfing-uncertainty-review]]
- Author index: [[authors/andrew-buskell]]

## Selected quotes

> "Clark's picture posits that the nervous system realizes a 'generative model', one that produces hypotheses about the spatiotemporal structure of sensory input … the result of an extended process of Bayesian inference."

> "Like some of his other books … this is an agenda-setter. It sets the stage for debates to come. The all-too-brief discussion of topics with massive primary and secondary literatures should be seen as promissory notes."
