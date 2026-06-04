---
type: blog-research
slug: predictive-processing
last_updated: 2026-06-04
---

# Research — Your brain was doing self-supervised learning first

Evidence base for the draft. Every item links to its vault source note (which
traces to the original). Items marked `_needs note_` are not yet in the vault —
read & deep-note before relying on them.

## Core claims (the PP mechanism)

- The brain is a **multi-layer probabilistic prediction machine**: higher levels predict the activity of lower levels, all the way down to the sensory periphery. — [[blogs/clark-surfing-uncertainty-precis]]
- Processing is **two streams**: bottom-up (precision-weighted sense data) and top-down (precision-weighted predictions), reconciled at every cortical layer by **Bayes' theorem**. — [[blogs/alexander-surfing-uncertainty-review]]
- It's the **prediction error** (precision-weighted mismatch), not raw sensation, that carries the information-processing load and signals what's "salient and newsworthy." — [[blogs/clark-surfing-uncertainty-precis]]
- Each layer's goal is to **minimize surprisal**. Three outcomes per layer: match → silence; low-precision data vs high-precision prediction → "cook the books" (rewrite sensation); high-precision conflict → surprisal alarm propagates up. — [[blogs/alexander-surfing-uncertainty-review]]
- **Perception = "controlled hallucination":** you experience the top-down generative model's hypotheses, *constrained* (not determined) by sense data. — [[blogs/alexander-surfing-uncertainty-review]], [[blogs/clark-surfing-uncertainty-precis]]
- Ascending the hierarchy = representing **probability distributions over increasingly temporally-extended trajectories** of sensory input — not richer mirror-images of the world. — [[blogs/buskell-surfing-uncertainty-review]]
- Lineage: Karl Friston, Geoffrey Hinton, David Knill, Alexandre Pouget. — [[blogs/buskell-surfing-uncertainty-review]]

## The "predict to learn" bridge to ML (the spine of the piece)

- Clark **explicitly** invokes large-corpus ML: trying repeatedly to predict the next word teaches you a surprising amount of grammar — the prediction task itself bootstraps world-knowledge. He places PP in a lineage running "all the way to recent work in deep learning." — [[blogs/clark-surfing-uncertainty-precis]]
- **Unsupervised learning, PP version:** keep generative models whose predicted sense data match observation; discard the rest. — [[blogs/alexander-surfing-uncertainty-review]]
- **Hyperpriors** (sensory synchronicity, object permanence) can be innate *or learned* — PP is "compatible with a pretty fricking blank slate." (Cited evidence: the 2003 Molyneux's-problem result that cross-modal links are *learned*, not intuitive.) — [[blogs/alexander-surfing-uncertainty-review]]
- ML analogue in the vault: **BYOL-Explore** uses a self-predictive latent loss as *both* the representation objective and the exploration bonus — prediction error doing double duty, exactly PP's "error = what's newsworthy." — [[papers/guo-2022-byol-explore]], [[topics/self-predictive-learning]]
- ML analogue: **JEPA** learns by predicting *embeddings* of masked/future observations in latent space rather than reconstructing pixels — PP's "predict the distal causes, not the raw signal." — [[topics/jepa]]
- ML analogue: **LeWorldModel** is a stable JEPA world model from pixels; "controlled hallucination + offline simulation" made literal as a generative world model that plans in latent space. — [[papers/maes-2026-leworldmodel]], [[topics/world-models]]

## Precision-weighting ≈ attention / learned uncertainty

- Both streams carry **precision estimates** (confidence), and Bayesian integration weights them accordingly. — [[blogs/alexander-surfing-uncertainty-review]]
- **Attention = the confidence interval / precision on predictions:** high attention up-weights bottom-up data; low attention lets top-down priors fill in (inattentional blindness — the gorilla; peripheral vision and the blind spot as top-down fill-in). — [[blogs/alexander-surfing-uncertainty-review]]
- Precision-weighting "modulates the influence of top-down priors relative to bottom-up stimuli"; normal cognition = the right *balance*. — [[blogs/buskell-surfing-uncertainty-review]]

## Action as prediction (active inference) ≈ control-as-inference

- The "But…": naively PP makes perception/representation paramount, but for an embodied creature **action** is what must be right. — [[blogs/clark-surfing-uncertainty-precis]]
- **Active inference:** to lift your arm, predict strongly that it's already lifted and let lower levels minimize proprioceptive prediction error. Friston (quoted): "motor commands have been replaced by … proprioceptive predictions." — [[blogs/alexander-surfing-uncertainty-review]]
- Perception and action share the same computation; only the **direction of fit** differs (shopping-list analogy). — [[blogs/alexander-surfing-uncertainty-review]]

## "Controlled hallucination cuts both ways" (the cautionary thread)

- Strong top-down priors can "cook the books" and *alter sensation* (PARIS IN THE THE SPRINGTIME; Asch conformity as books-cooked perception). — [[blogs/alexander-surfing-uncertainty-review]]
- But top-down stays **advisory** — it can't "steamroll arbitrary amounts of reality" unless pathological. Good guardrail for talking about LLM hallucination/confirmation. — [[blogs/alexander-surfing-uncertainty-review]]
- Pathology-as-imbalance: **autism** = over-precise priors / over-reliance on bottom-up → constant surprisal; **schizophrenia** = weak/aberrant priors → random prediction errors → delusions/hallucinations. (Same knob, opposite settings.) — [[blogs/alexander-surfing-uncertainty-review]], [[blogs/clark-surfing-uncertainty-precis]]
- Neurochemistry mapping (vivid, use sparingly): NMDA ≈ top-down, AMPA ≈ bottom-up, **dopamine ≈ precision/confidence**. — [[blogs/alexander-surfing-uncertainty-review]]

## The disanalogies / where to be careful (Buskell)

- PP is a **"mid-level organizational sketch,"** argued via cumulative inference-to-best-explanation, not a knock-down proof. — [[blogs/buskell-surfing-uncertainty-review]]
- **Hierarchy ambiguity** (the sharpest critique to import into ML): "moving up the hierarchy" is given three non-equivalent meanings — (a) bigger **spatiotemporal scope**, (b) sketchier high-level **gist/context**, (c) stronger **hyperprior**. These are *different architectural commitments*; conflating them is exactly the slippage that makes "brains do predictive coding, so our nets should too" arguments mushy. — [[blogs/buskell-surfing-uncertainty-review]]
- **Holton's "themata":** bottom-up vs top-down is partly an unfalsifiable framework commitment, not a settled empirical fact — committed feedforward theorists can always craft saving hypotheses. — [[blogs/buskell-surfing-uncertainty-review]]

## Quotes worth using

> "Biological brains are constantly active, trying to predict the streams of sensory stimulation before they arrive." — Andy Clark, via [[blogs/clark-surfing-uncertainty-precis]]

> "The key insight: the brain is a multi-layer prediction machine. All neural processing consists of two streams: a bottom-up stream of sense data, and a top-down stream of predictions." — Scott Alexander, via [[blogs/alexander-surfing-uncertainty-review]]

> "You're not seeing the world as it is, exactly. You're seeing your predictions about the world, cashed out as expected sensations, then shaped/constrained by the actual sense data." — via [[blogs/alexander-surfing-uncertainty-review]]

> "All that really matters (for any adaptive purpose that I can think of) is what you do, not what you see or perceive." — Andy Clark, via [[blogs/clark-surfing-uncertainty-precis]]

## Connections / contrasts (argument skeleton)

- **Agreement:** Clark (manifesto) and Alexander (explainer) converge on the same two-stream / error-minimization mechanism; Alexander supplies the ML-legible framing and the worked examples.
- **Tension to exploit:** Buskell's hierarchy critique is the honest counterweight — it's also *directly* an ML design question (what does "depth" buy: scope, abstraction, or prior strength?). This is the section that makes the post more than a hype piece.
- **Vault throughline:** PP → self-supervised prediction is the same throughline the [[topics/predictive-processing]] MOC draws to [[topics/jepa]] / [[topics/self-predictive-learning]] / [[topics/world-models]].

## Gaps / to-research (read & deep-note before leaning on these)

- **Active inference / Friston primary sources** — _needs note_. The motor/active-inference claims currently route through Alexander's secondary summary; for a strong RL ↔ active-inference section, read a primary active-inference reference (Friston 2010 "free-energy principle"; Da Costa et al. active-inference tutorial) and deep-note it.
- **Predictive coding ANNs** (Rao & Ballard 1999; recent predictive-coding-as-backprop-approximation work) — _needs note_. Would let the post connect PP to actual trainable predictive-coding nets, not just analogy.
- **Control-as-inference** (Levine 2018 tutorial) — _needs note_. The rigorous bridge for the "action = inference" claim on the ML side.
- Optional: a clean next-token-prediction-as-world-model reference to anchor the LLM analogy (the vault has diffusion/JEPA LMs but no canonical AR-next-token note yet).
