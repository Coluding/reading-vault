---
type: moc
topic: predictive-processing
last_updated: 2026-07-23
---

# Predictive Processing

The neuroscience/philosophy-of-mind framework — also called the **Bayesian brain**,
**predictive coding**, or (with action) **active inference** — holding that the
brain is a **hierarchical generative model** that constantly predicts its own
sensory stream, and that **prediction error** (precision-weighted mismatch), not
raw sensation, drives perception, learning, and action. Perception becomes
"controlled hallucination"; action becomes the fulfilment of proprioceptive
predictions. Lineage: Friston, Hinton, Knill, Pouget. This is the cognitive-science
taproot of the vault's self-supervised-prediction threads
([[topics/jepa]], [[topics/self-predictive-learning]], [[topics/world-models]]):
"predict your own input to bootstrap world structure" is the same move ML makes
with next-token and latent-prediction objectives.

This cluster collects three readings of Andy Clark's *Surfing Uncertainty*
(OUP, 2016), the only book-length treatment of PP — the manifesto, the accessible
explainer, and the scholarly critique.

## The book — Surfing Uncertainty (Andy Clark, 2016)

- [[blogs/clark-surfing-uncertainty-precis]] — **author's own précis** (The Brains Blog). The brain as a multi-layer probabilistic prediction machine; prediction-as-bootstrap of world knowledge (explicitly citing large-corpus ML); perception → understanding → imagination; the embodiment "But…" resolved by active inference. *Primary-source framing.*
- [[blogs/alexander-surfing-uncertainty-review]] — **Scott Alexander / SSC**, the accessible deep explainer. Two precision-weighted streams reconciled by Bayes at every layer; "cook the books" vs surprisal; the "explain everything" tour: attention, imagination, learning/hyperpriors, motor active inference, placebo, autism (over-precise priors), schizophrenia (weak priors). *Best ML-legible on-ramp.*
- [[blogs/buskell-surfing-uncertainty-review]] — **Andrew Buskell / BSPS**, the scholarly critique. PP as a "mid-level organizational sketch" argued via cumulative inference-to-best-explanation; isolates an ambiguity in what "moving up the hierarchy" means (spatiotemporal scope vs gist/context vs hyperprior); Holton's "themata" framing of the bottom-up-vs-top-down dispute. *The honest counterweight.*

## Papers

- [[papers/zhang-2026-learnable-novelty]] — **Learnable novelty** (Zhang & Levin): intelligence as maximizing the epiplexity a bounded observer can compress into a model; closed-form reservoir estimator ranks rule 110 top of all 88 ECA; MNIST probe 0.53→0.89 unsupervised; stable intrinsic RL reward (9/10 envs, zero collapses).

## Related topics

- [[topics/jepa]] — latent-space prediction; the ML analogue of perceptual prediction error
- [[topics/self-predictive-learning]] — BYOL-style self-prediction; prediction error as learning (and exploration) signal
- [[topics/world-models]] — "controlled hallucination" + offline simulation = generative world modelling
- [[topics/brain-inspired]] — predictive coding as a brain-derived architectural prior
