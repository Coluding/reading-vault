---
type: paper
title: "AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence"
authors: ["Jeff Clune"]
year: 2019
venue: arXiv
url: https://arxiv.org/abs/1905.10985
rw_id: "manual:clune-2019"
topics: [open-ended-learning, exploration, general-intelligence, meta-learning, position-paper]
priority: high
read_state: queued
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# AI-GAs: AI-generating algorithms

**TL;DR** — Clune's 2019 position paper proposing an alternative to the "manual AI approach" (hand-engineer pieces of intelligence and combine them later) — instead, build **AI-Generating Algorithms (AI-GAs)** that automatically discover the components of general AI through three pillars: (1) meta-learning architectures, (2) meta-learning the learning algorithms themselves, and (3) generating effective learning environments. The argument: just as deep learning replaced hand-engineered features with learned features, AI-GAs aim to replace the next layer of hand-engineering — the *whole stack* of architecture / learner / environment — with end-to-end learning.

> _Ingested manually as a citation chase from [[papers/jiang-2022-rethinking-exploration]] (cited as foundational for open-ended self-improving systems). Source: arXiv abstract page; not via Readwise. Only the abstract was fetched._

## Context & motivation

Clune frames general AI as the "most ambitious scientific quest in human history" and identifies the dominant paradigm as the **manual AI approach**:

> "The dominant approach in the machine learning community is to attempt to discover each of the pieces required for intelligence, with the implicit assumption that some future group will complete the Herculean task of figuring out how to combine all of those pieces into a complex thinking machine. I call this the 'manual AI approach'." — abstract

The argument structure: ML history has been a series of "let's remove the next layer of hand-engineering" moves — from hand-engineered features to learned features, from hand-tuned architectures to NAS, from hand-tuned hyperparameters to meta-learning. AI-GAs are the logical next step: remove hand-engineering from the *whole* stack of architecture, learning algorithm, and training environment.

## The three pillars (as named in the paper's metadata)

1. **Meta-learning architectures** — automate the search for neural architectures (NAS-style at scale).
2. **Meta-learning the learning algorithms themselves** — meta-learn the update rules / optimisers / loss surrogates, not just initialisations.
3. **Generating effective learning environments** — automate the curriculum / environment / dataset itself; tightly aligned with [[papers/jiang-2022-rethinking-exploration]]'s outer-loop framing.

The fetched abstract does not unpack each pillar's machinery — these are named at the abstract level and presumably expanded in the body.

## Method

_The abstract is short and does not describe a specific method. AI-GAs is a research-program proposal, not a single algorithm. Body method details are **not addressed by the source** at the abstract level._

## Experimental setup

_None — position paper, no experiments at the abstract level._

## Key results

_None at the abstract level — the paper is making a programmatic argument, not an empirical claim._

## Limitations

_Not addressed by the source_ at the abstract level.

[analyst's view] Limitations a reader would flag:
- Computational cost. Each of the three pillars is already compute-hungry on its own; doing all three jointly seems orders of magnitude harder.
- Specification problem. "Generating effective learning environments" sidesteps the question of *what makes an environment effective* — that's exactly the [[papers/jiang-2022-rethinking-exploration]] outer-loop problem, restated.
- Search-cost-vs-discovery-quality trade-off. NAS is already known to often discover only marginal improvements over hand-designed architectures; the same critique applies to meta-learning learning algorithms.

## Why it matters [analyst's view]

Three reasons this paper is worth holding in the vault:

1. **It names the next layer of automation.** The "manual AI approach" framing is rhetorically sharp and turns out to be predictive — by 2026, RLHF, synthetic data generation, autocurricula, self-play, and LLM-self-improvement loops are all instances of AI-GAs' third pillar in practice.

2. **It's the cleanest pre-LLM articulation of the "data and environment are the bottleneck" thesis.** Three years before [[papers/jiang-2022-rethinking-exploration]] formalised the outer/inner-loop distinction, Clune was already arguing that the data/environment generation pillar deserved equal investment with architecture/optimisation. Jiang 2022 is partly a refinement of this thesis with cleaner formalism.

3. **The "Why investigate now?" argument has aged well.** Reading 2019 Clune from 2026, every modern frontier-lab investment (synthetic data, evals, autocurricula, RLHF) is doing some version of AI-GA's third pillar. The proposal preceded the trend.

## Open questions

[analyst's view]
- **What does pillar 2 actually look like in practice?** Meta-learning the learning algorithm has fewer flagship results than pillars 1 (NAS) and 3 (autocurricula). What's the strongest 2026 evidence for it?
- **Cross-link to GRAM ([[papers/baek-2026-gram]]) and HRM ([[papers/wang-2025-hierarchical-reasoning-model]]):** these papers learn the *latent dynamics of reasoning* — could that be a version of pillar 2 for reasoning tasks?
- **Pillar 3 in LLM post-training:** RLHF and synthetic data are pillar-3 in practice. But these typically don't *also* run pillar 1 and pillar 2 jointly. Why? Is the AI-GA hypothesis that pillars compound multiplicatively, or are they tractable separately?

## Connections

- **Conceptual successor / refinement**: [[papers/jiang-2022-rethinking-exploration]] — Jiang et al. cite Clune's AI-GAs explicitly and build the outer/inner-loop formalism on top of this thesis.
- **Inner-loop counterpart**: [[papers/guo-2022-byol-explore]] — BYOL-Explore is a concrete instantiation of "exploration via novelty/curiosity" that fits inside AI-GAs' broader vision.
- **Topic MOCs**: [[topics/open-ended-learning]], [[topics/exploration]]
- **Author indices**: [[authors/jeff-clune]]

## Selected quotes

> "Perhaps the most ambitious scientific quest in human history is the creation of general artificial intelligence, which roughly means AI that is as smart or smarter than humans." — abstract

> "The dominant approach in the machine learning community is to attempt to discover each of the pieces required for intelligence, with the implicit assumption that some future group will complete the Herculean task of figuring out how to combine all of those pieces into a complex thinking machine. I call this the 'manual AI approach'. This paper describes another exciting path that ultimately may be more successful at producing general AI." — abstract

## Source caveats

- Ingested manually via citation chase, not Readwise.
- Only the arXiv abstract page was fetched. Body method, experiments, and any specific empirical claims are **not addressed by the source** at this triage depth. A re-read with the full PDF is required before citing specific section content.
