---
type: moc
topic: exploration
last_updated: 2026-05-22
---

# Exploration

Two distinct senses both treated under "exploration" in this vault: (1) the classical RL question of *what action to take next* under uncertainty (intrinsic-reward methods, curiosity, novelty search); (2) the broader, system-level question of *what data to learn from next* — including environment design, autocurricula, and the post-2022 framing that exploration is the bottleneck for general intelligence.

## Foundational position

- [[papers/jiang-2022-rethinking-exploration]] — Jiang, Rocktäschel, Grefenstette (Meta AI / UCL / Cohere, late 2022): position paper unifying exploration across SL and RL via a two-level outer/inner-loop framework; argues open-ended exploration is necessary for *increasingly general intelligence* (IGI). The conceptual map for the rest of the topic.

## Inner-loop / intrinsic-reward methods

- [[papers/guo-2022-byol-explore]] — BYOL-style self-predictive loss as joint world-model + intrinsic reward; solves DM-HARD-8 hard-exploration tasks without human demos.

## Related topics

- [[topics/open-ended-learning]]
- [[topics/reinforcement-learning]]
- [[topics/self-predictive-learning]]
- [[topics/world-models]]
