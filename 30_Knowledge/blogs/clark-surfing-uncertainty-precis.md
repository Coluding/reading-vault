---
type: blog
title: "Surfing Uncertainty: Prediction, Action, and the Embodied Mind (author's précis)"
author: "Andy Clark"
url: https://philosophyofbrains.com/2015/12/14/surfing-uncertainty-prediction-action-and-the-embodied-mind.aspx
rw_id: 01kt9he0tkkkyjr5a152nw4w1j
topics: [predictive-processing, self-predictive-learning, world-models]
priority: medium
read_state: deep
added: 2026-06-04
last_updated: 2026-06-04
---

# Surfing Uncertainty — author's précis (Andy Clark)

## TL;DR

Andy Clark's own guest-blog précis of his book *Surfing Uncertainty* (OUP, 2016),
arguing that the brain is, at its core, a **multi-layer probabilistic prediction
machine**: higher levels continually try to predict the activity of lower levels
all the way down to the sensory periphery, and it is the **prediction errors** —
not the raw sense data — that carry the information-processing load. He frames
"predictive processing" (PP) as the candidate *unifying* theory that finally gives
embodied cognitive science a principled account of what the biological brain
contributes.

## Context

Written as the first of a week-long series of guest posts on *The Brains Blog*
(Dec 2015) introducing the book. Clark — a philosopher of mind long associated
with embodied/extended cognition (*Being There*, the extended-mind thesis) — uses
it to recant his own earlier view that the brain is "just a rag-bag of different
tricks," and to argue that PP may be the few-core-principles story he'd previously
doubted existed.

## Core argument

The brain runs a **downward cascade of predictions** and an **upward cascade of
prediction errors**. Perceiving a coffee cup *is* the multi-level neural guess
that best cancels visual prediction error: the brain predicts the scene and lets
mismatch signals refine the guess until equilibrium. Three payoffs he stresses:

1. **Learning by prediction bootstraps world-knowledge.** He explicitly invokes
   large-corpus machine learning: trying repeatedly to predict the next word
   teaches you a surprising amount of grammar — the prediction task itself is the
   route to the structure. Multi-level prediction yields a multi-scale grip on the
   world (low levels: edges/lines; high levels: objects/events), with top-down
   predictions modulating even early visual areas (V1).
2. **Perception phases into understanding, and into imagination.** Run the same
   prediction machinery "offline" and you get imagination, mental simulation, and
   a bridge to simulation-based reasoning.
3. **Pathology as imbalance.** Delusions, hallucinations, drug effects, and
   autistic profiles are reframed as disturbances in the balance between top-down
   prediction and bottom-up evidence.

The post's twist (the "But…"): naively, PP makes *perception* (error reduction)
sound paramount and representational — but for an embodied creature it's **action**
that must be right, not representational fidelity. Clark previews the resolution
(active inference): PP is really about "efficiently translating energetic
stimulation into action," with action itself driven by predicting its own
proprioceptive consequences.

## Notable details

- The "strange architecture" framing (quoting Patrick Winston: information flows top-down, bottom-up, and sideways) — PP claims the downward flow's job is prediction.
- Lineage Clark places PP in: roots from early perception work through to "recent work in deep learning."
- The social/political edge (via Lisa Feldman Barrett): strong priors can make a police officer "see" a gun in a phone — top-down prediction has real-world stakes.

## Why it matters [analyst's view]

This is the cognitive-science taproot of a lot of what the vault already tracks.
The "predict your own sensory stream to bootstrap world structure" thesis is the
neuroscience sibling of **self-supervised prediction** in ML — next-token
prediction, and especially latent-prediction objectives like
[[topics/jepa]] / [[topics/self-predictive-learning]] (BYOL-Explore's
prediction-error-as-bonus is almost a literal PP move). "Perception as controlled
hallucination" + offline simulation is the cognitive gloss on generative
[[topics/world-models]]. And the action half — driving behaviour by predicting
proprioceptive consequences — is active inference, the control-as-inference cousin
of RL. Reading Clark alongside HRM / JEPA papers makes the brain-inspired framing
in the vault concrete rather than rhetorical.

## Connections

- Topic MOCs: [[topics/predictive-processing]]
- Related: [[topics/jepa]], [[topics/self-predictive-learning]], [[topics/world-models]], [[papers/guo-2022-byol-explore]] (prediction-error as intrinsic signal)
- Companion notes (same book): [[blogs/alexander-surfing-uncertainty-review]], [[blogs/buskell-surfing-uncertainty-review]]
- Author index: [[authors/andy-clark]]

## Selected quotes

> "Biological brains are constantly active, trying to predict the streams of sensory stimulation before they arrive. Systems like that are most strongly impacted by sensed deviations from their predicted states."

> "All that really matters (for any adaptive purpose that I can think of) is what you do, not what you see or perceive."
