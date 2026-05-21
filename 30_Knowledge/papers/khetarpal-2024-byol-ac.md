---
type: paper
title: "A Unifying Framework for Action-Conditional Self-Predictive Reinforcement Learning"
authors: ["Khimya Khetarpal", "Zhaohan Daniel Guo", "Bernardo Avila Pires", "Yunhao Tang", "Clare Lyle", "Mark Rowland", "Nicolas Heess", "Diana Borsa", "Arthur Guez", "Will Dabney"]
year: 2024
venue: arXiv
url: https://arxiv.org/abs/2406.02035
rw_id: 01ks5eqrn0a0s77qfqb2t0zmsv
topics: [self-predictive-learning, representation-learning-rl]
priority: medium
read_state: queued
relevance: ""
added: 2026-05-21
last_updated: 2026-05-21
---

# A Unifying Framework for Action-Conditional Self-Predictive RL (BYOL-AC)

**TL;DR** — Extends the BYOL-Π ODE theory of self-predictive representation learning to the action-conditional case (BYOL-AC), derives a third objective (BYOL-VAR) from the variance relation between BYOL-Π and BYOL-AC, and unifies all three under both a model-based (low-rank dynamics approximation) and model-free (value/Q/advantage) lens. Empirically BYOL-AC wins across linear and Deep RL settings.

## Problem setup

- "Learning a good representation is a crucial challenge for Reinforcement Learning (RL) agents. Self-predictive learning provides means to jointly learn a latent representation and dynamics model by bootstrapping from future latent representations (BYOL)" (abstract).
- The gap addressed: existing ODE theory assumes "the algorithm depends on a fixed policy (BYOL-Π); this assumption is at odds with practical instantiations of such algorithms, which explicitly condition their predictions on future actions" (abstract).
- Concrete environments / scale are _not addressed by the source_ beyond "linear function approximation and Deep RL environments" (abstract).

## Method

- Define and analyse **BYOL-AC**: "an action-conditional self-predictive objective … using the ODE framework, characterizing its convergence properties and highlighting important distinctions between the limiting solutions of the BYOL-Π and BYOL-AC dynamics" (abstract).
- A **variance equation** links the two representations: "this connection leads to a novel variance-like action-conditional objective (BYOL-VAR) and its corresponding ODE" (abstract).
- Two unifying lenses (abstract):
  - **Model-based**: "each objective is shown to be equivalent to a low-rank approximation of certain dynamics".
  - **Model-free**: relating objectives to "their respective value, Q-value, and advantage function".
- The exact ODEs, low-rank approximations, and architectural choices are _not addressed by the source_ at the abstract level.

## Results

- "Empirical investigations, encompassing both linear function approximation and Deep RL environments, demonstrate that BYOL-AC is better overall in a variety of different settings" (abstract).
- Specific benchmarks, magnitudes of improvement, and ablation details are _not addressed by the source_.

## Why it matters

[analyst's view] — This closes a real theory↔practice gap. The BYOL-Π analyses (Tang et al. 2022) implicitly assumed a fixed policy, but practitioners (including [[papers/guo-2022-byol-explore]]) condition predictions on actions, so the theoretical guarantees did not actually cover what was deployed. Mapping the three objectives to value / Q-value / advantage gives a principled selector: pick the representation loss whose limit equals the quantity downstream RL actually needs.

The variance-based BYOL-VAR is the part to watch — variance-style objectives have surfaced repeatedly in curiosity / disagreement methods, so the same loss appearing organically from BYOL theory is suggestive.

## Open questions

[analyst's view]
- Does BYOL-VAR yield a useful intrinsic-reward signal in the BYOL-Explore mould, or does it collapse?
- How does the model-based "low-rank approximation of dynamics" view connect to factorised world-model literature?
- Linear theory + deep RL empirics: where does the gap show up? Are there regimes where BYOL-Π still wins?

## Related

- [[papers/guo-2022-byol-explore]] — practical action-conditional BYOL agent; shared authors Guo, Pires, Tang. BYOL-AC theory directly addresses the regime BYOL-Explore operates in.
- [[topics/self-predictive-learning]]
- [[topics/representation-learning-rl]]
- [[authors/khimya-khetarpal]]
