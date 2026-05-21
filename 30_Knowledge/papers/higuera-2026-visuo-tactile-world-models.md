---
type: paper
title: "Visuo-Tactile World Models"
authors: ["Carolina Higuera", "Sergio Arnaud", "Byron Boots", "Mustafa Mukadam", "Francois Robert Hogan", "Franziska Meier"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2602.06001
rw_id: 01ks5mgnj18t4dc3s5f44n1fqr
topics: [world-models, robotics, tactile-sensing, multimodal-representation]
priority: high
read_state: queued
relevance: ""
added: 2026-05-21
last_updated: 2026-05-21
---

# Visuo-Tactile World Models

**TL;DR** — A multi-task world model (VT-WM) that fuses vision with tactile sensing to model the physics of contact for robot manipulation. The authors claim that grounding the latent dynamics in touch fixes vision-only failure modes — objects disappearing, teleporting, or violating physics under occlusion — yielding 33% better object permanence and 29% better motion-law compliance in autoregressive rollouts, and up to 35% higher zero-shot real-robot success on contact-rich tasks.

## Problem setup

- Target setting: "contact-rich manipulation tasks" where vision-only world models fail under "occlusion or ambiguous contact states" (abstract).
- Named failure modes the paper attributes to vision-only models: "objects disappearing, teleporting, or moving in ways that violate basic physics" (abstract).
- Two evaluation regimes named in the abstract:
  - **Imagination quality** — autoregressive rollouts scored for object permanence and laws-of-motion compliance.
  - **Zero-shot real-robot planning** — multi-step contact-rich tasks executed on physical hardware.
- Specific task suite, tactile sensor hardware, datasets, and baseline identities are _not addressed by the source_ (only the arXiv abstract was fetched).

## Method

- Core proposal: "multi-task Visuo-Tactile World Models (VT-WM), which capture the physics of contact through touch reasoning" (abstract).
- Trained "across a set of contact-rich manipulation tasks" (abstract) — multi-task, single shared model.
- Architecture (visual encoder, tactile encoder, fusion mechanism, dynamics head, planning procedure) is _not addressed by the source_ at the abstract level.
- Training objective and loss formulation are _not addressed by the source_.
- Planning procedure used for real-robot rollouts is _not addressed by the source_.

## Results

- **Imagination physics fidelity (autoregressive rollouts)**:
  - "33% better performance at maintaining object permanence" (abstract).
  - "29% better compliance with the laws of motion" (abstract).
  - Baseline against which these deltas are measured is _not addressed by the source_ (presumed vision-only ablation).
- **Zero-shot real-robot success rate**:
  - "Up to 35% higher success rates, with the largest gains in multi-step, contact-rich tasks" (abstract).
- **Few-shot adaptation**:
  - "Significant downstream versatility, effectively adapting its learned contact dynamics to a novel task and achieving reliable planning success with only a limited set of demonstrations" (abstract).
- Exact numbers, task-by-task breakdowns, and ablations are _not addressed by the source_.

## Ablations

_Not addressed by the source_ at the abstract level. The 33% / 29% / 35% deltas imply a controlled vision-only comparison exists in the paper, but the abstract does not name it.

## Limitations

_Not addressed by the source_ at the abstract level.

[analyst's view] Open questions a reader would flag from the abstract alone:
- Does VT-WM degrade gracefully when tactile is missing or noisy, or does fusion create a hard dependency?
- "33% better object permanence" — measured how? (Tracking metric? Frame-level detection? Hand-crafted physics violation rate?)

## Why it matters [analyst's view]

This paper sits in the small but growing intersection of *world models* and *embodied robot perception*. Two things stand out:

1. **The physics failure modes named are exactly what video world models get blamed for** — objects teleporting, vanishing, violating conservation laws. [[papers/joseph-2026-physics-video-world-models]] argues that even strong video models encode physics in a distributed, non-factorised way; VT-WM's pitch is essentially "you can't reason about contact physics from pixels alone in the first place — touch is load-bearing." That's a fundamentally different attack on the same problem.
2. **Few-shot adaptation from a multi-task base** is the part to watch operationally. If the contact-dynamics representation transfers to novel tasks with limited demos, this is a viable substrate for downstream robot policy learning — closer to a "foundation world model for manipulation" than a single-task system.

Worth pairing with the [[topics/world-models]] line: this is the first paper in the vault to make tactile-conditioning load-bearing rather than vision-conditioning.

## Open questions

[analyst's view]
- How is tactile actually fused — early concat, cross-attention, separate dynamics branch? Different choices have very different inductive biases for contact physics.
- What's the train/eval gap between simulation and real hardware? The 35% real-robot gain is the headline, but the abstract doesn't disclose how sim-to-real was handled.
- Does VT-WM's improvement come from the modality (touch as signal) or from the inductive bias (an explicit contact-physics module conditioned on touch)? Pure additive-signal vs. structural change has different implications for scaling.
- How does this compare to action-conditional latent dynamics models that don't use touch (e.g., the BYOL-AC family — [[papers/khetarpal-2024-byol-ac]])? Is the gain orthogonal or substitutional?

## Connections

- Contrasts with: [[papers/joseph-2026-physics-video-world-models]] — both confront physics failure in world models, but from opposite directions (representation analysis vs. modality augmentation).
- Topic MOCs: [[topics/world-models]]
- Author indices: [[authors/carolina-higuera]]

## Selected quotes

> "We introduce multi-task Visuo-Tactile World Models (VT-WM), which capture the physics of contact through touch reasoning." — abstract

> "VT-WM better understands robot-object interactions in contact-rich tasks, avoiding common failure modes of vision-only models under occlusion or ambiguous contact states, such as objects disappearing, teleporting, or moving in ways that violate basic physics." — abstract

> "In zero-shot real-robot experiments, VT-WM achieves up to 35% higher success rates, with the largest gains in multi-step, contact-rich tasks." — abstract
