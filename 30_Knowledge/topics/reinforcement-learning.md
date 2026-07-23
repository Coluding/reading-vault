---
type: moc
topic: reinforcement-learning
last_updated: 2026-07-23
---

# Reinforcement Learning

_Vault notes touching reinforcement learning._

## Notes
- [[papers/momennejad-2026-compositional-open-ended]] — This is a **theoretical / position paper** (no experiments) that proposes a formal object for open-ended intelligence built from a finite set of primi
- [[papers/jiang-2025-world4rl]] — **World4RL** is a two-stage framework that uses a **diffusion-based world model as a high-fidelity, frozen simulator** in which to refine an imitation
- [[papers/shi-2026-gpc-motor-control]] — GPC (Generative Pretrained Controllers) builds general-purpose, reusable controllers for physics-based character animation by borrowing the LLM recipe
- [[blogs/interlatent-ai-robotics]] — A first-principles tour of how modern "physical AI" robot policies are built, framed around the idea that a robot policy is just a function mapping ob
- [[papers/jiang-2022-rethinking-exploration]] — 
- [[papers/zuo-2026-qwen-agentworld]] — **Qwen-AgentWorld**: a native language world model as environment simulator for agent training — predicting next observations from history+action puts the model-based-RL loop inside a text generator.
- [[papers/chen-2026-actionable-simulators]] — position survey: a world model's value is decision-support (counterfactuals, intervention planning, long-horizon foresight), not rollout realism; argues for closed-loop decision-oriented evaluation.
- [[papers/ding-2024-world-models-survey]] — survey whose "implicit representation" branch covers model-based RL (Dreamer, TD-MPC2) as one of the two founding lineages of world models.
- [[papers/piriyakulkij-2025-poe-world]] — **PoE-World**: world model as an exponentially-weighted product of hundreds of LLM-synthesized Python expert programs; from <1000 demo frames, the only method scoring on Montezuma's Revenge (100 vs 0 for PPO@20M); zero-shot to Alt levels.
- [[papers/mazzaglia-2024-genrl]] — **GenRL** (NeurIPS 2024): frozen InternVideo2 video–language space aligned to a vision-only world-model latent via a learned connector; language/video-prompted tasks trained purely in imagination; 0.80 vs 0.70 best baseline over 35 reward-free tasks; data-free adaptation variant.
