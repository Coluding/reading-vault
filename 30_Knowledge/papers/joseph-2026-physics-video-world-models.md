---
type: paper
title: "Interpreting Physics in Video World Models"
authors: ["Sonia Joseph", "Quentin Garrido", "Randall Balestriero", "Matthew Kowal", "Thomas Fel", "Shahab Bakhtiari", "Blake Richards", "Mike Rabbat"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2602.07050
rw_id: 01ks5kaw3q3rjgtqj2njvhfbxz
topics: [interpretability, video-world-models, world-models]
priority: high
read_state: queued
relevance: ""
added: 2026-05-21
last_updated: 2026-05-21
---

# Interpreting Physics in Video World Models

**TL;DR** — Probes large video encoders for physical variables and finds a sharp intermediate-depth "Physics Emergence Zone" where physics becomes decodable. Direction is encoded as a high-dimensional population code with circular geometry, not as a factorised scalar — so modern video world models do not internally implement a classical physics engine, but their distributed representation is still sufficient for accurate physical prediction.

## Problem setup

- Core question (abstract): "whether video-based models need to rely on factorized representations of physical variables in order to make physically accurate predictions, or whether they can implicitly represent such variables in a task-specific, distributed manner."
- "Modern video world models achieve strong performance on intuitive physics benchmarks", but "it remains unclear which of these representational regimes they implement internally" (abstract).
- "First interpretability study to directly examine physical representations inside large-scale video encoders" (abstract).
- Exact encoder list, pretraining objectives, and benchmark suites are _not addressed by the source_ at the abstract level.

## Method

- Four probing tools (abstract):
  - **Layerwise probing** — decode physical variables from each layer.
  - **Subspace geometry** — characterise the linear/non-linear structure of physics representations.
  - **Patch-level decoding** — localise where in the visual field physics emerges.
  - **Targeted attention ablations** — test causal necessity of identified components.
- Variables decomposed include speed, acceleration, and motion direction (abstract).
- Specific probe architectures and ablation protocols are _not addressed by the source_.

## Results

- **Physics Emergence Zone**: "Across architectures, we identify a sharp intermediate-depth transition … at which physical variables become accessible. Physics-related representations peak shortly after this transition and degrade toward the output layers" (abstract).
- **Variable-dependent emergence**: "Scalar quantities such as speed and acceleration are available from early layers onwards, whereas motion direction becomes accessible only at the Physics Emergence Zone" (abstract).
- **Direction encoding is population-coded with circular geometry**: "direction is encoded through a high-dimensional population structure with circular geometry, requiring coordinated multi-feature intervention to control" (abstract).
- **Conclusion**: "Modern video models do not use factorized representations of physical variables like a classical physics engine. Instead, they use a distributed representation that is nonetheless sufficient for making physical predictions" (abstract).

## Why it matters

[analyst's view] — Two consequences land hard:

1. **Steering / counterfactual control is harder than people assume.** If you want to intervene on a video world model's prediction by altering "the direction variable", a single-feature edit won't work — you must coordinate a multi-feature edit respecting the circular population code. This is bad news for naive interpretability-based control schemes and good news for mechanistic-interpretability work that already takes population codes seriously (cf. SAE-style sparse-feature literature).
2. **The Physics Emergence Zone gives a concrete probe-layer prior.** Future probing studies on video models can start at the PEZ rather than scanning all depths — useful operationally and a hypothesis worth testing on other modalities (audio, robotics policies).

The "distributed but accurate" finding is also a quiet rebuke to the assumption that good predictive accuracy implies interpretable, modular internal variables.

## Open questions

[analyst's view]
- Is the Physics Emergence Zone universal across training objectives (predictive vs. contrastive vs. masked vs. generative) or specific to the encoder family tested?
- Does the circular geometry of direction emerge from data symmetries (the world is rotation-invariant) or from architectural inductive biases (rotary positional encodings)?
- How does the population-coded latent compare to what self-predictive RL world models like [[papers/guo-2022-byol-explore]] learn — is BYOL's latent equally non-factorised?
- The result is on encoders. Do decoder-side / generative video models show the same Physics Emergence Zone, or does the geometry shift?

## Related

- [[topics/interpretability]]
- [[topics/world-models]]
- [[topics/video-world-models]]
- [[authors/sonia-joseph]]
