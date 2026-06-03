---
type: blog
title: "What are the real problems of continual learning?"
author: "Andrew Lampinen"
url: https://infinitefaculty.substack.com/p/what-are-the-real-problems-of-continual
rw_id: 01kt7ncv3ksdy2134chas5pt7w
topics: [continual-learning, scaling-laws]
priority: medium
read_state: deep
added: 2026-06-03
last_updated: 2026-06-03
---

# What are the real problems of continual learning?

## TL;DR

Lampinen argues the continual-learning field spent decades attacking the
**wrong** problem. Catastrophic interference and loss of plasticity — its two
classic obsessions — turn out to be *largely artifacts of small models and hard
task boundaries*, and are substantially ameliorated by scale, pretraining,
normalization, and weight decay. The **real** remaining problem is **positive
transfer / cumulative learning**: getting a model to use its vast prior knowledge
to learn a genuinely new domain faster and to feed new knowledge back into old.
That, he claims, is where LLMs are still far weaker than humans.

## Context

A short essay (~2.3k words) on the *Infinite Faculty* Substack, written amid a
surge of interest in continual learning as "the area where the human–AI gap is
largest." Lampinen (a cognitive scientist) writes as someone re-framing a field
he thinks has been chasing the wrong targets, and ties the argument to his own
recent preprint on why scaling helps language models learn.

## Core argument

The piece walks the historical problem list and deflates each in turn:

1. **Catastrophic interference** (backward: new tasks wreck old ones) motivated
   replay (→ hippocampus/cortex complementary-learning theories) and
   weight-protection methods. But recent work shows old-task knowledge often
   *survives* in representations — interference is concentrated at **readout
   layers**, with earlier layers retaining high linear decodability, so tasks
   recover by retraining the output layer. "It may not be fundamental."
2. **Loss of plasticity** (forward: old learning impairs new learning) is real
   but **layer norm and weight decay substantially reduce it**, and some argue
   it's an artifact of *hard* task boundaries — continuous drift preserves
   plasticity.
3. **Scale + pretraining change the paradigm.** If LLMs suffered serious
   interference across pretraining → midtraining → SFT → RL, those pipelines
   "simply would not work." Wider models forget less (sparser, more orthogonal
   gradients); pretrained models forget less than from-scratch, and that benefit
   *grows* with scale. He notes a 1993 cog-sci paper already observed pretraining
   on related tasks reduces interference.

The load-bearing claim from his own preprint: larger LMs learn more *precisely
because* reduced interference + more memorization capacity let them preserve rare
structure until it's next needed.

## Notable details

- **The actual frontier is cumulative learning** — forward *and* backward
  *positive* transfer (a mathematician learns the next concept faster, and
  reinterprets earlier ones). LLMs, despite enormous task exposure, lack a
  universal recipe to transfer in-context knowledge *beyond* that context.
- Survey of partial fixes, all framed as bridges across the artificial
  in-context/parametric boundary: **data augmentation / "learn by thinking"**
  (distill self-generated inferences back into weights), **document retrieval**,
  **KV-cache compression/recomposition** (efficiency tricks that can integrate
  knowledge across corpora — cf. [[blogs/bansal-kv-cache]]), **textual memory**
  (self-notes, now standard in chat apps), and **context/self-distillation**
  (training the model to predict as if the context were present).
- Each "partially ameliorates but may not fundamentally solve" — you can't fit
  arbitrarily many documents/KVs/notes in context, nor distill from every context.
- **His deeper thesis:** biological learning systems aren't cleanly separated by
  timescale (synaptic plasticity spans ms→lifetime; episodic memory spans
  minutes→years). The fix may be to **remove the artificial hard boundary**
  between in-context and parametric learning, letting multiple learning systems
  operate continuously across all timescales.
- Caveat he flags: continual learning may not be *necessary* for AI to produce
  new insights — LLMs already appear to.

## Why it matters [analyst's view]

A useful corrective: it reframes "continual learning" away from the
forgetting-mitigation literature (much of which it argues scale already solved)
toward **transfer**, which aligns it with capability research rather than a
niche subfield. The "dissolve the in-context vs parametric boundary" framing is
the most provocative bit — it casts memory systems, KV-cache tricks, and
distillation not as separate hacks but as crude approximations of one continuous
learning substrate. Connects to the vault's [[topics/scaling-laws]] thread (scale
as the quiet solver) and to efficiency work like [[blogs/bansal-kv-cache]].

## Connections

- Topic MOCs: [[topics/scaling-laws]]
- Related blogs: [[blogs/bansal-kv-cache]] (KV-cache compression as a continual-learning lever)
- Author index: [[authors/andrew-lampinen]]

## Selected quotes

> "Continual learning research focused on small models … However, more recently researchers have found that larger models exhibit substantially less interference."

> "Maybe allowing multiple systems of learning to work together across every timescale, instead of having entirely separate systems at different timescales, is needed to enable the system to more effectively learn cumulatively."
