---
type: moc
topic: recursive-reasoning
last_updated: 2026-06-03
---

# Recursive Reasoning

Architectures and scaffolds that scale reasoning depth through *repeated computation*, in contrast to scaling reasoning through *longer generation* (CoT) or *bigger models*. Two distinct senses of "recursive" coexist in the literature: (1) **latent recursion** — recurrent neural architectures that refine a hidden state across many steps (HRM, GRAM, TRM, Looped/Universal Transformers); (2) **symbolic recursion** — agentic harnesses where an LLM spawns subagents inside a programmable environment (RLMs). Same word, different mechanism.

## Latent-recursion architectures

- [[papers/wang-2025-hierarchical-reasoning-model]] — foundational HRM: two recurrent modules at different timescales ($f_H$ slow, $f_L$ fast), DEQ-style 1-step gradient, deep supervision, ACT halting. 27M params, 1000 examples, solves Sudoku-Extreme / Maze-Hard / ARC-AGI without pretraining or CoT.
- [[papers/baek-2026-gram]] — generalises HRM into probabilistic multi-trajectory recursion via stochastic high-level updates and amortized variational inference; introduces width-based inference-time scaling and an unconditional generative interpretation.

## Looped transformers (weight-shared recurrence)

The "Looped/Universal Transformer" branch of latent recursion has grown into its
own cluster — see [[topics/looped-transformers]] for the full MOC. A 2026 wave
attacks it from three angles:

- [[papers/lee-2026-looped-diffusion-lm]] — **LoopMDM**: looping early-middle layers of a masked diffusion LM; 3.3× training-FLOP efficiency, +8.5 GSM8K, inference-time loop scaling.
- [[papers/deng-2026-lt2-looped]] — **LT2**: subquadratic (linear/sparse) attention makes looping scalable — "looping turns compute into context" (rank-$T$ memory / $O(Tw)$ receptive field).
- [[papers/chen-2026-training-free-looped]] — retrofits looping onto *frozen* checkpoints at inference via the forward-Euler/ODE view; no training, +2.64 pp MMLU-Pro on Qwen3-4B.

## Symbolic-recursion scaffolds

- [[blogs/biswas-rlm-deep-dive]] — practitioner deep-dive of Recursive Language Models: pass-by-reference agentic scaffold where the LLM operates inside a Python REPL with an `llm_query` function for spawning subagents; subagent outputs return as REPL variables, not LLM messages.

## Related topics

- [[topics/looped-transformers]]
- [[topics/latent-reasoning]]
- [[topics/brain-inspired]]
- [[topics/agentic-harnesses]]
- [[topics/inference-time-scaling]]
- [[topics/equilibrium-models]]
