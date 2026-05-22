---
type: paper
title: "Abandoning Objectives: Evolution Through the Search for Novelty Alone"
authors: ["Joel Lehman", "Kenneth O. Stanley"]
year: 2011
venue: "Evolutionary Computation 19(2):189–223"
url: https://doi.org/10.1162/evco_a_00025
rw_id: "manual:lehman-2011"
topics: [open-ended-learning, exploration, novelty-search, evolutionary-computation]
priority: high
read_state: queued
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# Abandoning Objectives: Evolution Through the Search for Novelty Alone

**TL;DR** — Lehman & Stanley's foundational **novelty search** paper. The proposal: instead of optimising an explicit objective (fitness function), search purely for *behavioural novelty* — accept any candidate that does something different from prior candidates, regardless of whether it "looks like progress" by any task metric. The headline claim: novelty search outperforms objective-based evolutionary methods on maze navigation and biped walking. The paradoxical lesson: *some problems are best solved by methods that ignore the objective*, because objective functions can actively misdirect search toward dead ends.

> _Ingested manually as a citation chase from [[papers/jiang-2022-rethinking-exploration]] (cited as foundational for "taming open-endedness"). Source: Crossref metadata + Semantic Scholar. The journal-version abstract is **elided by the publisher**; only paraphrased summary and two direct quotes are available without journal access._

## Context & motivation

The paper sits in the broader evolutionary-computation tradition where the canonical training signal is a *fitness function* — an explicit objective the candidate population is selected against. Lehman & Stanley's intervention: fitness functions can be *deceptive*. They can produce gradients that point at local optima while obscuring the path to better global solutions.

The two direct quotes available via Crossref capture the thesis:

> "Objective functions themselves may actively misdirect search toward dead ends."

> "Some problems are best solved by methods that ignore the objective."

The proposal: replace the fitness function with a **novelty metric** — a measure of how *behaviourally different* a candidate is from previously-seen candidates, with no reference to whether the candidate is "closer to the goal."

## Method (as inferable from public metadata)

_Full method-section detail is **not addressed by the source** without journal-PDF access._ The paraphrased Crossref / Semantic Scholar summary states only that the method "searches for behavioural novelty" — replacing the fitness function in an otherwise standard evolutionary algorithm. The well-known novelty-search recipe (from the broader literature):

1. Define a *behaviour characterisation* — a low-dimensional vector summarising what a candidate *did* (e.g. for a maze, the final position; for biped walking, distance + gait features).
2. Maintain a **novelty archive** of behaviour characterisations seen so far.
3. Score each candidate by **distance to nearest neighbours** in the archive — high distance = novel.
4. Select for novelty, *not* for objective performance.

The above is the standard novelty-search formulation as it appears in subsequent literature; the exact formulation in the Lehman/Stanley 2011 paper requires the journal PDF to verify.

## Experimental setup (as named in metadata)

Two benchmark tasks named:
- **Maze navigation** — evolve neural-network agents to navigate mazes; deceptive because local fitness gradients can point into cul-de-sacs.
- **Biped walking** — evolve a bipedal controller; deceptive because intermediate gaits look bad even if they're stepping stones to good gaits.

Exact configurations are **not addressed by the source** without journal-PDF access.

## Key results

[from metadata summary]
- "Novelty-based approaches can outperform conventional methods on maze navigation and biped walking tasks."
- Specific quantitative deltas, runs, and statistical analyses are **not addressed by the source**.

## Ablations / variants

_Not addressed by the source_ without journal-PDF access. Subsequent literature (which I cannot cite as this paper) discusses quality-diversity hybrids, MAP-Elites, etc. — but those would be follow-up papers, not part of this triage.

## Limitations

_Not addressed by the source_ at the metadata level.

[analyst's view] Limitations a reader would flag:
- **The "behaviour characterisation" is hand-designed.** Novelty search dodges the explicit-objective problem by introducing an implicit one: someone must specify what counts as "behavioural difference." For maze navigation that's easy (final position); for open-ended domains it's the same problem in disguise.
- **No formal claim that novelty search globally outperforms objective search.** The paper's contribution is *existence*: there exist problems where novelty wins. The scope of that "exist" remains an open question.
- **Scaling.** Novelty archives grow without bound; nearest-neighbour distance computation gets expensive in high dimensions.

## Why it matters [analyst's view]

Three reasons this is worth a vault entry despite being a 2011 paper:

1. **It is the canonical empirical demonstration that objective optimisation can hurt.** The 643-citation count reflects this — novelty search became a touchstone in evolutionary computation and trickled into RL via curiosity / count-based exploration. The "objectives can deceive" finding is the conceptual ancestor of every modern intrinsic-motivation method, including [[papers/guo-2022-byol-explore]]'s prediction-error-as-reward.

2. **The "no objective at all" extreme is rhetorically important.** Most modern systems use *some* objective. Novelty search demonstrates that even *zero* explicit objective can outperform — which means the role of objective specification in learning is subtler than most ML practice assumes. This frames the "what is the right loss function?" question correctly.

3. **It is the empirical leg of the Stanley/Lehman open-endedness program** that [[papers/jiang-2022-rethinking-exploration]] credits as foundational. POWERPLAY ([[papers/schmidhuber-2013-powerplay]]) provided the *algorithmic framework* for open-ended learning; novelty search provides the *empirical demonstration* that abandoning explicit objectives can work.

## Open questions

[analyst's view]
- **What's the modern descendant of novelty search?** Quality-diversity methods (MAP-Elites) are direct successors; in RL, count-based exploration and curiosity-driven RL (RND, ICM, BYOL-Explore) are spiritual cousins.
- **Is novelty search compatible with gradient-based learning?** Evolutionary methods are gradient-free; novelty search's "distance to nearest archive entry" doesn't obviously translate into a differentiable loss. The closest gradient-based analogues are diversity regularizers and counterfactual-novelty rewards.
- **The "abandoning objectives" framing vs LLM RLHF.** RLHF is *all objective* (reward model). Is there an RLHF-with-novelty story? Some recent work on response-diversity preserves resemble this idea.
- **Re-read with full PDF needed.** Specific algorithmic details, statistical results, and the precise behaviour characterizations chosen for maze and biped tasks all need the journal text.

## Connections

- **Foundational ancestor**: cited by [[papers/jiang-2022-rethinking-exploration]] as foundational for taming open-endedness.
- **Algorithmic framework cousin**: [[papers/schmidhuber-2013-powerplay]] — POWERPLAY's "simplest unsolved problem" search is conceptually parallel to novelty search's "most novel candidate" selection.
- **Position-paper context**: [[papers/clune-2019-ai-gas]] — Clune's AI-GAs framework relies on novelty / exploration as foundational ingredients.
- **Modern intrinsic-reward analogue**: [[papers/guo-2022-byol-explore]] — BYOL-Explore's prediction-error-as-intrinsic-reward is the deep-RL descendant of novelty-as-signal.
- **Topic MOCs**: [[topics/open-ended-learning]], [[topics/exploration]], [[topics/evolutionary-computation]]
- **Author indices**: [[authors/joel-lehman]], [[authors/kenneth-stanley]]

## Selected quotes

> "Objective functions themselves may actively misdirect search toward dead ends." — Crossref metadata description

> "Some problems are best solved by methods that ignore the objective." — Crossref metadata description

## Source caveats

- **Ingested manually via citation chase, not Readwise.** No Readwise rw_id; source is the Crossref-published metadata description plus Semantic Scholar metadata.
- **The journal-version abstract is elided by the publisher (MIT Press) and not available without access to Evolutionary Computation 19(2).** All claims above attributed to this paper are limited to (i) title and bibliographic metadata, (ii) two direct quotes obtainable from public metadata, (iii) the publicly-known *fact* that the paper introduces novelty search on maze and biped tasks. Method-section specifics, ablation details, numerical results, and limitations as stated by the authors are **not addressed by the source** at this triage depth.
- **Verification step**: anyone citing specific claims from this paper should fetch the journal PDF (DOI: 10.1162/evco_a_00025) first.
