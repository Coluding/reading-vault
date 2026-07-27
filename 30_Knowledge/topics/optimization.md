---
type: moc
topic: optimization
last_updated: 2026-07-27
---

# Optimization

_Vault notes touching optimization._

## Notes
- [[papers/kumar-2026-supervised-memory-training]] — The paper introduces **Supervised Memory Training (SMT)**, a way to train nonlinear RNNs *without* backpropagation through time (BPTT)
- [[blogs/haegele-magnitude-direction]] — The post introduces **Magnitude-Direction Decoupling (MD)**: constrain each weight matrix's *direction* to a fixed-norm sphere while giving it a separ
- [[papers/knight-2026-riemannian-low-rank]] — A careful, honest study of **Riemannian optimization for rank-factored weight matrices** ($W = AB^\top$), treating the factorization as an implementat
- [[papers/degroot-2023-scenario-motion-planning]] — **Safe Horizon MPC**: nonconvex scenario optimization turns an intractable joint chance constraint into $S$ sampled hard constraints; sample size derived offline from $(\epsilon,\beta,\bar{n})$, support estimated online from active sets.

