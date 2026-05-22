---
type: moc
last_updated: 2026-05-22
status: living

---

# Dashboard — Recent

Most recent items added to the vault, regardless of category. Refreshed by
`/process-inbox`.

## This week

### Papers
- [[papers/jiang-2022-rethinking-exploration]] — position paper (Meta/UCL/Cohere, late 2022): exploration is the bottleneck for general intelligence; proposes the outer/inner-loop generalized-exploration framework unifying SL and RL.
- [[papers/pao-huang-2026-flux-matching]] — generalizes score matching to the full Fokker–Planck-consistent family of generative vector fields; turns the vector field into a design choice; faster mixing + interpretable fields + structured dynamics at 2–4× DSM training cost.
- [[papers/wang-2025-hierarchical-reasoning-model]] — foundational HRM (Sapient Intelligence): two recurrent modules at slow/fast timescales, DEQ-style 1-step gradient; 27M params, 1000 examples solve Sudoku-Extreme / Maze-Hard / ARC-AGI without pretraining or CoT.
- [[papers/baek-2026-gram]] — extends HRM into probabilistic multi-trajectory recursion via stochastic latent transitions + amortized variational inference; introduces width-based inference-time scaling; 99.7% N-Queens 8×8 vs HRM 78.7%.
- [[papers/yang-2026-replaid-continuous-diffusion]] — closes the AR-vs-continuous-diffusion compute gap from 64× to 20× via protocol alignment; SOTA 22.1 PPL on OpenWebText among continuous DLMs; shows ELBO-variance noise schedule = linear cross-entropy.
- [[papers/guo-2022-byol-explore]] — BYOL-style self-predictive loss as joint world-model + intrinsic reward; solves DM-HARD-8 without demos.
- [[papers/khetarpal-2024-byol-ac]] — ODE theory for action-conditional BYOL representations; unifies BYOL-Π / BYOL-AC / BYOL-VAR.
- [[papers/joseph-2026-physics-video-world-models]] — interpretability of physics in video encoders; identifies the Physics Emergence Zone and a circular population code for motion direction.
- [[papers/higuera-2026-visuo-tactile-world-models]] — multi-task world model fusing vision + tactile; claims tactile grounding fixes physics-violation failure modes and yields up to 35% higher zero-shot real-robot success.
- [[papers/tong-2026-beyond-language-modeling]] — controlled from-scratch native-multimodal pretraining (Transfusion + MoE); finds a scaling asymmetry (vision more data-hungry than language) and world-modeling capabilities emerge from generic training.
- [[papers/huang-2026-semantic-tube-prediction]] — generalises JEPA to language via the "Geodesic Hypothesis"; claims 16× LLM data-efficiency over Chinchilla on NL-RX-SYNTH.
- [[papers/maes-2026-leworldmodel]] — first stable end-to-end JEPA from pixels; two-term loss (prediction MSE + SIGReg) replaces EMA / stop-gradient / multi-term-loss orthodoxy; plans up to 48× faster than DINO-WM.

### Blogs
- [[blogs/biswas-rlm-deep-dive]] — practitioner deep-dive on Recursive Language Models: pass-by-reference agentic scaffold (LLM operates inside a Python REPL with `llm_query` for subagents); 60M-character Lex Fridman transcript in 4 minutes for $0.20.

## Last week

_Nothing yet._

## Older

_Nothing yet._

---

## How this dashboard works

This file is rewritten at the end of every `/process-inbox` run. It shows:

- **This week**: items added in the last 7 days, grouped by category.
- **Last week**: items added 7–14 days ago.
- **Older**: rolled-up by month.

For deeper queries see the other dashboards in `30_Knowledge/_MOCs/`:

- `dashboard-deep-read.md` — items marked `priority: high` and `read_state: queued`
- `dashboard-orphans.md` — notes with no inbound links (potentially mis-filed)
