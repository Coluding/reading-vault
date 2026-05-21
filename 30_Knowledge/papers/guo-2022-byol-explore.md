---
type: paper
title: "BYOL-Explore: Exploration by Bootstrapped Prediction"
authors: ["Zhaohan Daniel Guo", "Shantanu Thakoor", "Miruna Pîslar", "Bernardo Avila Pires", "Florent Altché", "Corentin Tallec", "Alaa Saade", "Daniele Calandriello", "Jean-Bastien Grill", "Yunhao Tang", "Michal Valko", "Rémi Munos", "Mohammad Gheshlaghi Azar", "Bilal Piot"]
year: 2022
venue: arXiv
url: https://arxiv.org/abs/2206.08332
rw_id: 01ks5jt54te29qhtdk1ew77dpb
topics: [exploration, self-predictive-learning, world-models]
priority: medium
read_state: queued
relevance: ""
added: 2026-05-21
last_updated: 2026-05-21
---

# BYOL-Explore: Exploration by Bootstrapped Prediction

**TL;DR** — A curiosity-driven exploration agent in which a single BYOL-style self-predictive loss in latent space jointly trains the world representation, the world-dynamics model, and an exploration policy; the prediction error doubles as the intrinsic reward. Authors claim it solves DM-HARD-8 without human demonstrations and achieves superhuman performance on the ten hardest Atari exploration games.

## Problem setup

- Targets "curiosity-driven exploration in visually-complex environments" (abstract).
- Two evaluation regimes named in the abstract:
  - **DM-HARD-8** — "a challenging partially-observable continuous-action hard-exploration benchmark with visually-rich 3-D environments" where "prior work could only get off the ground with human demonstrations" (abstract).
  - The "ten hardest exploration games in Atari" (abstract).
- Further benchmark details, baselines, and ablations are _not addressed by the source_ (only the arXiv abstract was fetched).

## Method

- Learns "a world representation, the world dynamics, and an exploration policy all-together by optimizing a single prediction loss in the latent space with no additional auxiliary objective" (abstract).
- The single loss is BYOL-style — i.e. predicting one's own latent — applied to world-model rollouts. The intrinsic reward is the magnitude of that latent prediction error.
- Implementation/architectural details (encoder family, target network update, action conditioning specifics) are _not addressed by the source_ at the abstract level.

## Results

- On DM-HARD-8, "we solve the majority of the tasks purely through augmenting the extrinsic reward with BYOL-Explore's intrinsic reward, whereas prior work could only get off the ground with human demonstrations" (abstract).
- On Atari, "achieves superhuman performance on the ten hardest exploration games … while having a much simpler design than other competitive agents" (abstract).
- Numeric scores, sample-efficiency curves, and comparisons to RND/ICM/Go-Explore are _not addressed by the source_.

## Why it matters

[analyst's view] — Collapsing representation learning and intrinsic reward into the same objective removes a coordination problem: traditional curiosity methods (RND, ICM) keep a novelty model separate from the agent's representation, which can drift. The DM-HARD-8 result — beating an evals regime that "previously required human demonstrations" — is the strongest signal that this collapse is not just elegant but unlocks tasks that were previously intractable from pure RL.

This is also the practical anchor for the BYOL-Π / BYOL-AC theory line: BYOL-Explore conditions on actions in practice, which is exactly the gap [[papers/khetarpal-2024-byol-ac]] analyses theoretically.

## Open questions

[analyst's view]
- Self-predictive losses are notoriously prone to collapse. The abstract claims "no additional auxiliary objective" — what stabilises the latent here? (Target network? Predictor head asymmetry? Stop-gradients?)
- Does the intrinsic reward decay smoothly as the world model fits, or does it require explicit annealing?
- How does this stack against Go-Explore on Montezuma's Revenge / Pitfall specifically?

## Related

- [[papers/khetarpal-2024-byol-ac]] — action-conditional self-predictive RL theory; the practical setting BYOL-Explore operates in.
- [[topics/self-predictive-learning]]
- [[topics/world-models]]
- [[topics/exploration]]
- [[authors/zhaohan-daniel-guo]]
