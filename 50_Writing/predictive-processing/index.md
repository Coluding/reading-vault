---
type: blog-draft
title: "Your brain was doing self-supervised learning first"
slug: predictive-processing
status: gathering
audience: "ML practitioners & technically-literate generalists"
angle: "Predictive processing — the neuroscience theory that the brain is a hierarchical prediction machine — quietly anticipated the core bet of modern ML: that predicting your own input is enough to learn the world. Read PP as a spec, and next-token prediction, JEPA, and active inference all fall out of one idea."
target_length: "1800-2800 words"
topics: [predictive-processing, jepa, self-predictive-learning, world-models]
source_notes:
  - "[[blogs/clark-surfing-uncertainty-precis]]"
  - "[[blogs/alexander-surfing-uncertainty-review]]"
  - "[[blogs/buskell-surfing-uncertainty-review]]"
  - "[[topics/predictive-processing]]"
  - "[[papers/guo-2022-byol-explore]]"
  - "[[papers/maes-2026-leworldmodel]]"
created: 2026-06-04
last_updated: 2026-06-04
---

# Your brain was doing self-supervised learning first

## Thesis / hook

Predictive processing (PP) says the brain is a hierarchical generative model whose
only job is to predict its own sensory stream and minimize the error. That's the
*same bet* modern ML made — next-token prediction, JEPA latent prediction,
BYOL-style self-prediction, control-as-inference. If you read PP as an
architecture spec rather than a neuroscience curiosity, a surprising amount of the
last decade of self-supervised learning looks like rediscovery.

## Why write this

Most ML people meet "the Bayesian brain" as hand-wavy inspiration. But PP is
unusually *mechanistic* — two precision-weighted streams, Bayesian reconciliation
per layer, surprisal minimization, action as fulfilled prediction. Laying it next
to JEPA / next-token / active inference makes the analogy precise instead of
vibes, and the *disanalogies* are where the interesting research questions live.
The vault already has the three canonical PP readings deep-noted plus the ML-side
notes, so the research is in hand.

## Key messages

- **One objective, many faces.** "Predict your own input to bootstrap world
  structure" underlies grammar-from-next-word, JEPA latent prediction, and PP
  perception alike.
- **Precision-weighting ≈ learned uncertainty / attention.** PP's confidence-
  weighting of streams is a concrete cognitive analogue of uncertainty-aware
  objectives and attention as confidence.
- **Action is just prediction with a different direction of fit** (active
  inference) — the cognitive twin of control-as-inference RL.
- **"Controlled hallucination" cuts both ways.** Strong priors that help
  perception are the same mechanism behind hallucination/confirmation — in brains
  *and* large generative models.
- **The disanalogies matter** (Buskell's hierarchy ambiguity): "deeper = bigger
  scope" vs "deeper = more abstract gist" vs "deeper = stronger prior" are
  *different* architectural commitments. Don't smuggle them together.

## Working titles

- "Your brain was doing self-supervised learning first"
- "Predictive processing for ML people"
- "Prediction is all you need (the neuroscience version)"
- "Your brain is a JEPA"

## Source notes

- [[blogs/clark-surfing-uncertainty-precis]] — the manifesto (Andy Clark)
- [[blogs/alexander-surfing-uncertainty-review]] — the ML-legible explainer (Scott Alexander)
- [[blogs/buskell-surfing-uncertainty-review]] — the scholarly critique (Buskell)
- [[topics/predictive-processing]] — the MOC tying it to the vault's ML threads
- [[papers/guo-2022-byol-explore]] — prediction error as both representation loss and exploration bonus
- [[papers/maes-2026-leworldmodel]] — JEPA world model; "controlled hallucination" made literal

## Status notes

- **2026-06-04** — Created from the predictive-processing cluster. `research.md`
  populated from the three Surfing Uncertainty notes + ML-side notes. `draft.md`
  has an outline. Next: decide whether to lead with the JEPA analogy or the
  next-token analogy; write the opening. Open question: how hard to push the
  active-inference ↔ RL section without overclaiming (I'm not deep-noted on active
  inference / Friston primary sources yet — see `research.md` gaps).
