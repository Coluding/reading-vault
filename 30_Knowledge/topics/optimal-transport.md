---
type: moc
topic: optimal-transport
last_updated: 2026-06-22
---

# Optimal Transport

Optimal transport (OT) studies the cost-minimizing way to move one distribution
onto another — Monge/Kantorovich formulations, the Wasserstein metric, Brenier's
theorem (the squared-cost optimal map is the gradient of a convex potential), and
entropic relaxations (Sinkhorn). In this vault OT shows up as the *unifying lens*
for generative sampling: straight, non-crossing transport paths are what make
few-step generation possible, and several 2026 works recast familiar machinery
(flow matching, attention) as OT problems.

## Notes
- [[papers/malnick-2026-designing-ot-flows]] — sidesteps intractable OT by **designing a prior whose identity coupling is already OT-optimal** (low-frequency projections); straightens flow-matching trajectories with no OT solver.
- [[blogs/jiha-autoregression-vs-diffusion]] — reframes AR vs diffusion as two solutions to *one* OT problem: AR = Knothe–Rosenblatt rearrangement (triangular, exact, ordering-biased); diffusion/flow = learned unconstrained Brenier map.
- [[papers/litman-2026-attention-priors]] — recasts attention as **entropic** OT with an implicit uniform prior; positional encodings are heuristic approximations of an EOT-derived prior term.

## Related topics
- [[topics/flow-matching]]
- [[topics/diffusion-models]]
- [[topics/generative-models]]
