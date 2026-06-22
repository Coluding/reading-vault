---
type: moc
topic: distillation
last_updated: 2026-06-22
---

# Distillation

Compressing the work of a slow/expensive generative process (typically a
many-step diffusion or flow sampler, or a stronger teacher model) into a fast
student that produces comparable samples in one or a few steps. In the vault this
is tightly coupled to the few-step-generation literature: consistency models,
MeanFlow, shortcut models, distribution-matching distillation (DMD/VSD), and
reflow.

## Notes
- [[blogs/dieleman-diffusion-integral]] — survey of the few-step / flow-map literature (consistency models, MeanFlow, shortcut models, Align Your Flow) — distillation as learning a flow map that jumps across time.
- [[blogs/accelerated-diffusion-tutorial]] — CVPR 2026 "FastGen" tutorial; distillation as one of the three pillars of accelerated generation.
- [[papers/cai-2026-mode-mean-seeking]] — mode-seeking head distilled from a frozen short-video teacher via reverse-KL DMD/VSD, aligning each sliding window of the student's rollout — no extra long-video data needed.
- [[papers/malnick-2026-designing-ot-flows]] — *complements* distillation: modifies the coupling rather than the objective, and is shown to compose with the one-step MeanFlow framework.

## Related topics
- [[topics/flow-matching]]
- [[topics/diffusion-models]]
- [[topics/generative-models]]
