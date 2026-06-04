---
type: blog-draft-body
slug: predictive-processing
status: outlining
last_updated: 2026-06-04
---

# Your brain was doing self-supervised learning first

> Draft. Outline is complete; opening is a starter you can rip up. Every factual
> claim should trace to `research.md` (and through it to a vault note). Sections
> marked _[your call]_ are decisions to make while writing.

## Outline

1. **Hook** — The ML field spent the 2010s discovering that "just predict the next
   thing" is shockingly powerful. Neuroscience got there first, and wrote down the
   spec. Meet predictive processing.
2. **The mechanism in one section** — Two streams (bottom-up sense data, top-down
   prediction), reconciled by Bayes at every layer, minimizing surprisal.
   Perception as "controlled hallucination." _(Source: Alexander, Clark.)_
3. **The bridge: predict-to-learn** — Clark *himself* uses next-word prediction to
   explain how the brain bootstraps structure. Map it onto next-token, then onto
   JEPA (predict embeddings, not pixels), then onto BYOL-Explore (error as both
   signal and curiosity). _(Source: Clark, JEPA/BYOL notes.)_
4. **Precision-weighting = attention / learned uncertainty** — Confidence on each
   stream; attention as confidence interval; the gorilla. _(Source: Alexander.)_
5. **Action is just prediction (active inference)** — Move by predicting you've
   already moved; control-as-inference parallel. Flag the honesty caveat: I'm
   leaning on a secondary summary here _[your call: how hard to push]_.
   _(Source: Alexander; gaps in research.md.)_
6. **The cut-both-ways section** — controlled hallucination is also the mechanism
   of hallucination/confirmation, in brains and LLMs alike; autism vs schizophrenia
   as the same precision knob at opposite settings. _(Source: Alexander, Clark.)_
7. **Where the analogy breaks (and why that's the interesting part)** — Buskell's
   hierarchy ambiguity, reframed as an ML design question: does network depth buy
   *scope*, *abstraction*, or *prior strength*? Don't conflate them. _(Source:
   Buskell.)_
8. **Takeaway** — PP isn't proof our nets are brains; it's a well-specified prior
   on what "learning by prediction" can be. Read it as a spec, mine the
   disanalogies for research.

---

## Draft

### 1. Hook _(starter — rewrite freely)_

Sometime in the last decade, machine learning collapsed onto a single bet: don't
label the world, just predict it. Next token, next frame, next masked patch — pour
in enough self-supervised prediction and structure falls out the other side. It
worked so well it stopped feeling like a choice.

Here's the thing nobody mentions in the methods section: neuroscience made the
same bet first, and unlike us, it wrote down the spec. It's called **predictive
processing**, and once you read it as an architecture rather than a theory of the
brain, an uncomfortable amount of modern ML starts to look like rediscovery.

> _Continue from the outline. Pull claims + quotes from `research.md`._

### 2. The mechanism

_[write — two-stream / Bayes-per-layer / surprisal / controlled hallucination]_

### 3. Predict to learn

_[write — Clark's next-word example → next-token → JEPA → BYOL-Explore]_

### 4. Precision-weighting

_[write]_

### 5. Action as inference

_[write — keep honest per research.md gaps]_

### 6. Cuts both ways

_[write]_

### 7. Where it breaks

_[write — Buskell's hierarchy ambiguity as an ML design question]_

### 8. Takeaway

_[write]_
