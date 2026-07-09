---
type: moc
topic: flow-matching
last_updated: 2026-07-08
---

# Flow Matching

Flow matching learns a time-dependent velocity field $v_\theta(x_t,t)$ that
transports a tractable prior $p_0$ to the data $p_1$ along (conditionally)
straight interpolation paths, trained by regressing onto the conditional velocity
$x_1-x_0$. The 2026 cluster in this vault converges on a single theme: the
*coupling* between prior and data, and the *straightness* of the resulting
trajectories, are the levers for fast (few-/one-step) generation — and they can
be attacked from the prior side, the objective side, or by distillation.

## Foundational / tutorial
- [[blogs/dieleman-diffusion-integral]] — Sander Dieleman's unifying tutorial on **flow maps** (predict any point on a noise→data path from any other), organized via three consistency rules (compositionality, Lagrangian, Eulerian); maps the 2024–26 few-step-sampling literature onto that grid.
- [[blogs/flow-based-llms-intro]] — Floor Eijkelboom's intro to **flow-based language models**: softmax + cross-entropy is exactly the Variational-Flow-Matching objective on the simplex; flows are distillable into one/two-step Categorical Flow Maps (unlike discrete diffusion).

## Recent
- [[papers/porcher-2026-flowwm]] — **FlowWM** is a stochastic visual world model that runs **flow matching directly inside the high-dimensional feature space of a frozen pretrained enco
- [[blogs/interlatent-ai-robotics]] — A first-principles tour of how modern "physical AI" robot policies are built, framed around the idea that a robot policy is just a function mapping ob
- [[papers/malnick-2026-designing-ot-flows]] — **design the prior so its identity coupling to the data is OT-optimal** (low-frequency image projections); >2× straighter trajectories without solving OT; composes with MeanFlow.
- [[papers/cai-2026-mode-mean-seeking]] — decoupled mean-seeking (FM) + mode-seeking (distribution-matching) heads for fast long video generation; FM head learns coherence, discarded at inference.

## Related topics
- [[topics/optimal-transport]]
- [[topics/diffusion-models]]
- [[topics/distillation]]
- [[topics/generative-models]]
