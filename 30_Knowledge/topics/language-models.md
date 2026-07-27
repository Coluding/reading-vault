---
type: moc
topic: language-models
last_updated: 2026-07-27
---

# Language Models

_Vault notes touching language models._

## Notes
- [[papers/bayat-2026-tapered-language-models]] — Transformers (and their modern relatives) allocate the same MLP width to every layer
- [[papers/zou-2026-latent-thought-flow]] — Latent Thought Flow (LTF) makes LLM reasoning happen inside the model's continuous hidden space instead of decoded chain-of-thought tokens, and — unli
- [[papers/zuo-2026-qwen-agentworld]] — **Qwen-AgentWorld** (Qwen team): the first *native* language world model — next-observation prediction as a pre-training objective, making an LLM double as an environment simulator for general agents.
- [[papers/cheng-2026-dspark]] — **DSpark** (DeepSeek-AI): how these models are actually *served* — semi-autoregressive speculative decoding plus load-aware verification scheduling, validated on live DeepSeek-V4 traffic.
- [[papers/piriyakulkij-2025-poe-world]] — **PoE-World**: world model as an exponentially-weighted product of hundreds of LLM-synthesized Python expert programs; from <1000 demo frames, the only method scoring on Montezuma's Revenge (100 vs 0 for PPO@20M); zero-shot to Alt levels.
