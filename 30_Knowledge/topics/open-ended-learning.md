---
type: moc
topic: open-ended-learning
last_updated: 2026-05-22
---

# Open-Ended Learning

The line of work arguing that *learning systems should continually invent or discover their own problems* — not just solve given ones. Open-ended learners produce systems whose capabilities expand indefinitely because the supply of training tasks expands with them. The vault contains both the foundational position/algorithmic proposals (2011–2019) and the modern refinement (Jiang 2022).

## Foundational

- [[papers/lehman-2011-novelty-search]] — empirical demonstration that abandoning the explicit objective and selecting for behavioural *novelty* can outperform objective-based search on deceptive problems (maze navigation, biped walking). The empirical leg of the Stanley/Lehman open-endedness program.
- [[papers/schmidhuber-2013-powerplay]] — algorithmic framework for an agent that continually searches the (new-task, solver-modification) joint space, accepting any modification that provably solves all prior tasks plus a new one. The "self-curriculation as Kolmogorov-complexity search" formulation.
- [[papers/clune-2019-ai-gas]] — proposes three pillars (architecture meta-learning, learning-algorithm meta-learning, environment generation) as the alternative to the "manual AI approach." The cleanest pre-LLM statement of the "data and environment generation is the next layer of automation" thesis.

## Modern refinement

- [[papers/jiang-2022-rethinking-exploration]] — generalized exploration as a two-level outer/inner loop that unifies SL and RL. Cites all three foundational papers above; provides the formalism around "increasingly general intelligence (IGI)."

## Related topics

- [[topics/exploration]]
- [[topics/reinforcement-learning]]
- [[topics/general-intelligence]]
- [[topics/intrinsic-motivation]]
- [[topics/evolutionary-computation]]
