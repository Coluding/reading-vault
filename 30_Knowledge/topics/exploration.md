---
type: moc
topic: exploration
last_updated: 2026-07-23
---

# Exploration

Two distinct senses both treated under "exploration" in this vault: (1) the classical RL question of *what action to take next* under uncertainty (intrinsic-reward methods, curiosity, novelty search); (2) the broader, system-level question of *what data to learn from next* — including environment design, autocurricula, and the post-2022 framing that exploration is the bottleneck for general intelligence.

## Foundational position

- [[papers/jiang-2022-rethinking-exploration]] — Jiang, Rocktäschel, Grefenstette (Meta AI / UCL / Cohere, late 2022): position paper unifying exploration across SL and RL via a two-level outer/inner-loop framework; argues open-ended exploration is necessary for *increasingly general intelligence* (IGI). The conceptual map for the rest of the topic.
- [[papers/clune-2019-ai-gas]] — proposes "AI-generating algorithms" with three pillars including automatic generation of learning environments; pre-LLM articulation of the "data/environment is the bottleneck" thesis.
- [[papers/schmidhuber-2013-powerplay]] — algorithmic framework for self-invention of training problems via search over (new-task, solver-modification) pairs ordered by conditional computational complexity.

## Outer-loop / self-curriculation

- [[papers/lehman-2011-novelty-search]] — abandons explicit objectives in favour of selecting for behavioural novelty alone; foundational empirical demonstration that "objectives can mislead."

## Inner-loop / intrinsic-reward methods

- [[papers/guo-2022-byol-explore]] — BYOL-style self-predictive loss as joint world-model + intrinsic reward; solves DM-HARD-8 hard-exploration tasks without human demos.
- [[papers/zhang-2026-learnable-novelty]] — **Learnable novelty** (Zhang & Levin): intelligence as maximizing the epiplexity a bounded observer can compress into a model; closed-form reservoir estimator ranks rule 110 top of all 88 ECA; MNIST probe 0.53→0.89 unsupervised; stable intrinsic RL reward (9/10 envs, zero collapses).

## Related topics

- [[topics/open-ended-learning]]
- [[topics/reinforcement-learning]]
- [[topics/self-predictive-learning]]
- [[topics/world-models]]
