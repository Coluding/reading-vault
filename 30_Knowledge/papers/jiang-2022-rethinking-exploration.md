---
type: paper
title: "General Intelligence Requires Rethinking Exploration"
authors: ["Minqi Jiang", "Tim Rocktäschel", "Edward Grefenstette"]
year: 2022
venue: arXiv
url: https://arxiv.org/abs/2211.07819
rw_id: 01ks7dsvmb3vs8xy2bxca0ghd3
topics: [exploration, open-ended-learning, reinforcement-learning, general-intelligence, position-paper]
priority: high
read_state: queued
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# General Intelligence Requires Rethinking Exploration

**TL;DR** — A position paper (Meta AI / UCL / Cohere, late 2022) arguing that exploration is the bottleneck for general intelligence, and that the current treatment of exploration is fundamentally wrong on two fronts: supervised learning *ignores* it (treats data as static), and RL *under-scopes* it (only explores inside a static simulator). The authors propose **generalized exploration** as a two-level outer/inner-loop framework that unifies SL and RL — the outer loop continually expands the *data generator* itself (new datasets, new simulators, environment design), and the inner loop performs prioritized training on the current generator. They frame this as a necessary condition for **increasingly general intelligence (IGI)** — systems that, over time, surpass non-IGI agents in relative generality.

## Context & motivation

The paper opens with the framing thesis (§Introduction):

> "We are at the cusp of a transition from 'learning from data' to 'learning what data to learn from' as a central focus of artificial intelligence (AI) research. While the first-order learning problem is not completely solved, large models under unified architectures, such as transformers, have shifted the learning bottleneck from how to effectively train our models to how to effectively acquire and use task-relevant data."

The argument structure:
- Most large-model progress is in *optimization* (transformers, supervised learning, scaling laws).
- The remaining gap to generality is about *which data the model trains on*.
- Therefore the bottleneck has moved from learning-as-optimization to **exploration-as-data-curation**.

The paper is a *position* paper (not an empirical one), but it's careful to formalise its claims rather than just gesture at them.

### A relative definition of generality

To avoid the "what is AGI" trap, the authors define generality *relatively*:

> "Model A is more general than model B relative to a task set T if and only if A performs above a threshold level (e.g. that of a minimally viable solution) in more tasks in T than B, while at least matching the performance of B on all other tasks in T on which B meets the threshold."

A system showing *continual improvement* in relative generality is an **increasingly general intelligence (IGI)** — and the paper's thesis is that IGI requires open-ended exploration:

> "Increasingly general intelligence arises if and only if facilitated by open-ended exploration, which continually broadens the agent's capabilities."

## The two limits

### Limit 1 — Supervised learning ignores exploration

> "Models trained in this way will exhibit blindspots reflective of the missing information in their training data and suffer from covariate shifts and concept drift due to the non-stationarity of the real world."

The structural argument: all finite offline datasets are *incomplete* (the world has infinite facts) and *stationary* (datasets don't update themselves). A virtual assistant trained on a static conversational corpus falls out of date with culture and language; an LLM trained on a 2024 web crawl knows nothing of 2026.

### Limit 2 — RL under-scopes exploration

> "Current simulators predominantly mirror the limitations of a static, finite dataset in SL, and as such, RL agents share the same failure points as SL systems when deployed in the wild."

RL acknowledges exploration as a principal objective — but the exploration is *inside* a fixed simulator. Even NetHack-style parameterised environments can only span a bounded task distribution. From the paper's perspective, this is "the same failure mode as SL with extra steps."

The diagnosis:

> "The problem afflicting both classes of learning algorithms reduces to one of insufficient exploration: SL, largely trapped in the offline regime, fails to perform any exploration, while RL, limited to exploring the interior of a static simulation, largely ignores the greater expanse of possibilities that the simulation cannot express."

## Method — Generalized Exploration

The paper's framework (§4, Fig. 1) decomposes exploration into two loops:

### Outer loop — data-generator expansion
Continually expand the *space* from which training data is drawn. In SL terms: collect new training data (online or offline). In RL terms: search for *new simulator settings* — possibly generate new simulators (environment design, procedural content generation, automatic curriculum design).

### Inner loop — prioritized training
Given the current data generator, train the model with priorities over the generator's distribution. In SL: active sampling. In RL: prioritized experience replay, importance sampling.

The unifying claim: **both SL and RL fit the same outer/inner-loop pattern**. The only difference is that SL's outer loop is data collection and RL's outer loop is environment design.

> "This open-ended exploration process defines a new data-seeking outer-loop that continually expands the data generator used by the inner loop learning process, which itself may use more limited forms of exploration to optimally sample from this data generator."

### Zone of Proximal Development

The paper grounds the "what data should the outer loop seek?" question in developmental psychology:

> "Such exploration ideally seeks new challenges at the boundary of the agent's capabilities, focusing learning on tasks for which the current agent has the most potential to improve — analogous to Vygotsky's Zone of Proximal Development in developmental psychology."

The right exploration targets challenges that are *informative* — past the agent's current frontier but not too far past.

## Experimental setup & results

_Position paper — no empirical experiments._ The paper builds a conceptual framework and uses recurring examples (the virtual assistant) to test the framework's claims.

## Connections to existing methods

The paper organizes prior work under the generalized-exploration lens:
- **Curriculum learning** — prior-defined difficulty ordering; one inner-loop pattern.
- **Open-ended evolution** — Schmidhuber, Clune, Stanley/Lehman novelty search; closer to the outer-loop ideal but not yet integrated with modern deep learning.
- **PCG / unsupervised environment design (UED)** — Dennis et al.'s PAIRED / Jiang's own POET-style work; the outer-loop machinery the paper implicitly endorses.
- **Active learning in SL** — outer-loop exploration in SL, but typically scoped to a fixed dataset.
- **Continual learning** — addresses *retraining* on new data but doesn't address *finding* it.

The paper credits prior open-endedness work (Schmidhuber, Clune, Stanley, Lehman) and positions itself as a *consolidation*, not a first proposal.

## Limitations

[analyst's view] — the paper is explicit about its position-paper status. Open questions left for the reader:

- **No empirical validation.** The framework is conceptual; it predicts but does not demonstrate that generalized-exploration-based training will produce more general agents.
- **The outer loop is hard.** "Find a useful new simulator setting" is the central technical problem and the paper doesn't solve it. UED methods are gestured at but the work of actually realising open-ended-self-improvement remains open.
- **No formal generalisation bound.** The relative-generality definition is operational but doesn't yield a generalisation bound on out-of-distribution task performance.
- **Static-task-set assumption.** The relative-generality definition still requires a task set *T*. For genuinely open-ended settings, T is itself growing — the definition stretches but doesn't break, and the paper acknowledges this.

## Why it matters [analyst's view]

This paper is the cleanest articulation I've seen of the "data is the bottleneck" argument that has become consensus by 2026. Three things land:

1. **The unification of SL and RL exploration is non-obvious and right.** Most ML papers either treat exploration as an RL-only concept or treat SL as a "no exploration needed" case. The outer/inner decomposition makes clear that SL has been doing implicit, *bad* exploration (web crawling) all along — and that "what to crawl next" is the same problem as "what environment setting to train on next."

2. **It anticipated 2024-2026 trends.** The paper is from late 2022, before the "post-training" / "synthetic data" / "evals-as-data" wave. Re-reading it now: every concern raised (concept drift, covariate shift, static-dataset incompleteness) is now a multi-billion-dollar industry problem. The "generalized exploration" framing maps almost directly to today's practices in synthetic data generation, RLHF data collection, model self-improvement.

3. **The IGI framing has surprising philosophical bite.** Defining generality *relatively* sidesteps "is AGI achievable" debates and turns the question into "is the system continually expanding its relative-generality frontier?" That's empirically tractable and converts AGI from a binary milestone into a *gradient* worth optimizing.

The paper is foundational for the **open-ended learning** line of work — it's not the first proposal of these ideas (Schmidhuber, Clune, Stanley get credit) but it's the clearest map of how the ideas relate to modern ML practice. Worth flagging as a *background* read for any work on autocurricula, RLHF data design, or self-improving systems.

## Open questions

[analyst's view]
- **Does the outer/inner decomposition scale to LLM post-training?** RLHF iteration is implicitly an outer-loop expansion (prompts selected by humans / models). Is there a principled way to plug in the generalized-exploration framework here?
- **What's a tractable "Zone of Proximal Development" proxy at LLM scale?** Curiosity / surprise / disagreement / value-of-information all exist for small RL; their LLM analogues are less developed.
- **How do you evaluate IGI?** The relative-generality definition needs a stable task set *T*. For LLMs, "the set of tasks humans care about" is itself moving — eval design for IGI is its own subfield now.
- **Connection to [[papers/guo-2022-byol-explore]]** — BYOL-Explore is a concrete *inner-loop* exploration method that solves the curiosity-driven exploration problem in static simulators. Jiang et al. would say "this is necessary but not sufficient." Worth working through how BYOL-Explore-style intrinsic rewards interact with outer-loop environment design.
- **Connection to [[papers/baek-2026-gram]]** — GRAM's width-based inference-time scaling is a *sampling* answer to "how do I find more reasoning trajectories"; Jiang et al.'s thesis would say sampling is not enough, you need *new problems*, not just new trajectories. A clean conceptual contrast.

## Connections

- **Direct conceptual cousin**: [[papers/guo-2022-byol-explore]] — Jiang et al. and BYOL-Explore are from the same 2022 era and both address exploration; BYOL-Explore is the *inner-loop* answer (intrinsic reward for curiosity in a static sim); Jiang et al. argue the *outer loop* (the sim itself) is the bigger problem.
- **Background for**: any work on open-ended learning, autocurricula, environment design, synthetic data generation.
- **Tonally adjacent**: [[papers/tong-2026-beyond-language-modeling]] — the "scaling asymmetry" finding (vision more data-hungry than language) implicitly motivates better outer-loop exploration for visual data.
- **Topic MOCs**: [[topics/exploration]], [[topics/open-ended-learning]], [[topics/reinforcement-learning]]
- **Author indices**: [[authors/minqi-jiang]]

## Selected quotes

> "We are at the cusp of a transition from 'learning from data' to 'learning what data to learn from' as a central focus of artificial intelligence (AI) research." — §Introduction

> "Increasingly general intelligence arises if and only if facilitated by open-ended exploration, which continually broadens the agent's capabilities." — §Introduction

> "The problem afflicting both classes of learning algorithms reduces to one of insufficient exploration: SL, largely trapped in the offline regime, fails to perform any exploration, while RL, limited to exploring the interior of a static simulation, largely ignores the greater expanse of possibilities that the simulation cannot express." — §Introduction

> "Such exploration ideally seeks new challenges at the boundary of the agent's capabilities, focusing learning on tasks for which the current agent has the most potential to improve — analogous to Vygotsky's Zone of Proximal Development in developmental psychology." — §Introduction
