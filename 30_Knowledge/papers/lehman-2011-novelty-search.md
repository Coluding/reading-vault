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
read_state: skimmed
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# Abandoning Objectives: Evolution Through the Search for Novelty Alone

**TL;DR** — Lehman & Stanley's foundational **novelty search** paper. The proposal: replace fitness-driven evolutionary search with a *novelty-driven* one — score each candidate by how *behaviourally different* it is from previously-seen candidates (mean distance to $k$-nearest neighbours in a behaviour space, supplemented by an archive of past novel behaviours), and select on novelty alone, *ignoring* the objective. Demonstrated empirically on **deceptive maze navigation** and **biped walking** (with NEAT-evolved neural controllers): novelty search reliably finds solutions where objective-based search gets stuck in local optima created by misleading fitness gradients. The paradoxical thesis: *some problems are best solved by methods that ignore the objective*, because fitness functions can actively misdirect search toward dead ends. The paper is the conceptual ancestor of Quality-Diversity (MAP-Elites, Innovation Engines, POET) and a sibling of intrinsic-motivation methods in deep RL.

> _Ingested manually as a citation chase from [[papers/jiang-2022-rethinking-exploration]] (cited as foundational for "taming open-endedness"). Source: full paper PDF via swarthmore.edu mirror of Evolutionary Computation 19(2). This revision reflects a full-paper read, not metadata-only triage._

## Context & motivation

The paper sits inside the evolutionary-computation tradition where the canonical training signal is a *fitness function* — an explicit objective the population is selected against. Lehman & Stanley's intervention names a structural failure of this paradigm: **fitness functions can be deceptive**. They can produce gradients that point at local optima while obscuring the path to better global solutions.

The core thesis quotes:

> "Objective functions themselves may actively misdirect search toward dead ends."

> "Some problems are best solved by methods that ignore the objective."

The paper grounds *deception* in Goldberg's earlier foundational deceptiveness work and argues it is endemic to non-trivial fitness landscapes. The proposal: replace the fitness function with a **novelty metric** — how *behaviourally different* a candidate is from prior candidates, with no reference to whether the candidate is "closer to the goal."

## Method — the novelty search algorithm

The algorithm runs inside an otherwise standard evolutionary loop (the paper uses **NEAT** — NeuroEvolution of Augmenting Topologies — for the neural-controller experiments). The substitution is the *selection signal*.

### Behaviour characterisation

A candidate's behaviour is described by a low-dimensional vector that summarises *what it did*, not what it is. Domain-dependent, hand-designed:

- **Maze**: the final $(x, y)$ position of the agent at the end of the episode.
- **Biped walking**: the sequence of positions the biped's body visits during a fixed time horizon (variants test lengthened sequences and reduced precision).

Behaviour characterisation is *deliberately simple*. The maze case is striking: rather than reward "did you reach the goal," novelty search rewards "did you end up somewhere new" — yet this is enough to solve the maze.

### Novelty metric

Given a population and an archive of past behaviours, the **novelty score** of an individual $i$ is the mean distance to its $k$-nearest neighbours in behaviour space, considering both the current population and the archive:

$$\rho(i) = \frac{1}{k} \sum_{j=1}^{k} \text{dist}(b(i), b(\mu_j))$$

where $b(i)$ is the behaviour characterisation of $i$ and $\mu_j$ are its $k$-nearest neighbours.

Higher $\rho$ means the candidate is more isolated in behaviour space — more novel.

### Archive maintenance

Candidates whose novelty score exceeds an admission threshold are added to a persistent archive. The archive carries memory of "places we've been" across generations, preventing the population from re-discovering already-explored regions.

### Selection rule

Standard NEAT selection, but using $\rho(i)$ as the score in place of fitness. No explicit objective enters anywhere in the loop.

The result: pressure to *spread into unexplored regions of behaviour space*, with no pressure toward any goal.

## Experimental setup

### Maze navigation

NEAT-evolved neural-network controllers receive sensor input (distances to walls, direction to goal) and output motor commands. Mazes are constructed to be *deceptive*: there are dead ends close to the goal where fitness-based search gets stuck, because moving away from the goal looks like worse fitness even when it's the only path forward.

Variants tested (Sections 5–6):

- **Simple mazes** — basic case where fitness already works.
- **Mazes with walls removed** — direct test of whether fitness's apparent advantage holds when deception is reduced.
- **"Conflated" mazes** — designed to probe how behaviour characterisation choices affect novelty search.

Baseline: fitness-based NEAT, where fitness = (inverse) distance from final position to the goal.

### Biped walking

NEAT-evolved neural controller in a physics simulator. Fitness baseline: distance traveled. The deception story: in biped walking, early gaits that don't solve walking still produce *behaviourally diverse* locomotive attempts, which can be stepping stones to functional walking — but objective-based search prunes these as low-fitness.

### Variants of behaviour characterisation

Section 6.4 explicitly varies the behaviour-characterisation design — lengthening trajectories, reducing precision — as a hyperparameter ablation. This is one of the paper's most useful contributions: it demonstrates that the behaviour-characterisation choice is *the* design lever for novelty search.

## Key results

- **Novelty search outperforms fitness-based search on the deceptive maze**, including success rates and convergence speed. The result is robust to maze configuration variants.
- **Maze with walls removed**: novelty search's relative advantage is largely *unaffected* — meaning novelty search isn't winning *because* fitness is being deceived in some particular way, it's winning because of structural properties of the search.
- **Biped walking**: novelty search discovers walkers that fitness-based methods fail to find within comparable evaluation budgets, including some behaviourally distinct locomotion strategies that objective-based search prunes early.
- **Archive size remains tractable** (Section 6.2). Despite the open-ended-sounding "save everything novel" rule, the archive doesn't blow up — the admission threshold and behaviour-space structure keep it manageable.

Specific numerical comparisons (success rates, generations-to-solution, archive sizes) are reported in the paper's results tables.

## Ablations / variants

Section 6.4's behaviour-characterisation ablations are the most informative. Key findings:

- **Coarse behaviour characterisations work surprisingly well.** Even the simplest "final $(x,y)$" descriptor for the maze is enough to drive novelty search to success.
- **Trajectory length matters in biped.** Longer trajectories preserve more behavioural information but increase descriptor dimensionality; the paper finds a sweet spot rather than a monotone improvement.
- **Threshold tuning** affects archive growth and exploration rate.

Section 9.5 discusses "the arrow of complexity" — the qualitative observation that novelty-driven evolution tends to discover increasingly complex behaviours over time, mirroring biological evolution. This is not formalised as a theorem but is offered as an empirical pattern.

## Deception — the conceptual core

Section 2.1 builds the deception argument. The shape:

- A fitness function induces a gradient in genotype space.
- That gradient is reliable *only locally*; it can point away from globally good solutions if the landscape has misleading local optima.
- In domains where the goal is far from the starting region (mazes, biped walking, open-ended evolution), this is endemic, not exceptional.
- Removing the explicit objective and selecting on novelty *side-steps* the gradient-direction problem: novelty pressure can't be deceived by the same misleading local structure because it doesn't reference the goal at all.

==The conceptual move is closer to *random* search than to optimisation — but with the crucial addition of the archive, which prevents re-exploration and produces directed coverage of behaviour space.==  --> the archive is the important part here!!

## When novelty search works vs. fails (Section 9)

Lehman & Stanley are explicit about the boundaries of their claim.

**Where novelty search excels:**
- Behaviour space is well-chosen (i.e. behavioural differences are meaningful for the task).
- Objective-based search faces severe deception.
- The behaviour space is low-dimensional enough that novelty distance is meaningful.

**Where it struggles or fails:**
- Behaviour-space design requires human insight; in domains where you don't have a good descriptor, novelty search reduces to noise. Section 9.2 explicitly says *domain-independent open-endedness remains unsolved* — novelty search dodges the explicit-objective problem at the cost of introducing an implicit one (someone has to choose the behaviour space).
- In *high-dimensional* behaviour spaces, nearest-neighbour distance becomes uninformative.
- Novelty search alone doesn't *guarantee* solution discovery — it explores; whether a useful candidate appears in the archive is not guaranteed.

This is one of the paper's strongest features: it does not over-claim. The empirical demonstration is "novelty search wins on these tasks"; the conceptual claim is "objective-blind search is a legitimate paradigm worth investigating," not "ignore objectives forever."

## Minimal-criteria variants

The paper doesn't extensively formalise "minimal criteria novelty search" (MCNS) — that's developed in follow-up work — but the structural ingredient is present: the *archive admission threshold* is a minimal-criteria mechanism. Only behaviours sufficiently distant from existing archive members are retained. This is the seed that grows into MCNS and into the constraint-based novelty filtering used by later QD methods.

## Philosophical / open-endedness implications

Section 9.3 ("Novelty Search and Natural Evolution") draws the connection to biology. The argument: biological evolution did not optimise an explicit fitness function over geological time; it produced open-ended complexity by something closer to novelty-driven dynamics. Section 9.5 ("The Arrow of Complexity") is the rhetorical apex — novelty-driven evolution produces increasing complexity as an emergent property, not a designed-in target.

The paper cites Gould and others on evolutionary contingency, framing novelty search as a *computational substrate for studying open-ended evolution* — not just a clever ML trick. This is the angle that makes the paper a touchstone for the Stanley-Lehman open-endedness program that runs through "Why Greatness Cannot Be Planned" (popular treatment) and culminates in the QD literature.

## Limitations (paper's own + analyst's view)

Paper's own:
- Behaviour-space design is domain-specific; no general recipe.
- Archive cost grows with run length, even if slowly.
- No formal claim that novelty search globally beats objective search — the result is *existence* of domains where novelty wins, not universal dominance.
- High-dimensional behaviour spaces remain problematic.

[analyst's view] Additional limitations a 2026 reader would flag:
- **The "no objective at all" framing is a foil; in practice, hybrids dominate.** Subsequent QD work (MAP-Elites, NSLC) blends novelty and objective signals. Pure novelty search is a clean conceptual demonstration but rarely the practical sweet spot.
- **Behaviour characterisation is the new objective specification problem.** The paper dodges "what is the fitness?" by introducing "what is the behaviour descriptor?" — which is the same problem in different vocabulary. This is sharply visible in modern domains (LLM diversity) where "behaviour space" is much harder to define than $(x, y)$.
- **No gradient-based version.** Novelty search is structurally gradient-free. The deep-RL analogues (count-based exploration, RND, ICM, BYOL-Explore) are gradient-friendly but conceptually less pure — they reward *novelty as predicted by a model*, not novelty by definition.

## Why it matters [analyst's view]

Three reasons, after the full read:

1. **It is the canonical empirical demonstration that objective optimisation can hurt.** Before this paper, "deception" was a theoretical worry in evolutionary computation. After this paper, it was a *demonstrated phenomenon* with a clean alternative. Every modern intrinsic-motivation method — count-based exploration, RND, ICM, [[papers/guo-2022-byol-explore]] — inherits the conceptual move: prediction-error / novelty as a primary training signal.

2. **The "no objective at all" extreme is rhetorically important.** Most modern systems use *some* objective. Novelty search demonstrates that even *zero* explicit objective can outperform — which means the role of objective specification is subtler than most ML practice assumes. This frames the "what is the right loss function?" question correctly: sometimes the right answer is "don't have one."

3. **It is the empirical leg of the Stanley/Lehman open-endedness program.** POWERPLAY ([[papers/schmidhuber-2013-powerplay]]) provides the *algorithmic framework* for open-ended self-curriculation; novelty search provides the *empirical evidence* that abandoning explicit objectives produces good results in deceptive domains. [[papers/clune-2019-ai-gas]] pillar 3's QD descendants (MAP-Elites, POET, Innovation Engines) all directly extend the novelty-search idea.

## Open questions

[analyst's view]
- **Modern descendant of novelty search.** Quality-Diversity (MAP-Elites and family) are direct successors; in RL, count-based exploration and curiosity-driven RL (RND, ICM, BYOL-Explore) are spiritual cousins. Is there a clean 2026-vintage "novelty search for transformer training data" paper? Probably yes — worth a citation chase.
- **Is novelty search compatible with gradient-based learning?** The closest gradient-based analogues are diversity regularisers, counterfactual-novelty rewards, and discriminator-based diversity bonuses. None feel as clean as the original.
- **The "abandoning objectives" framing vs LLM RLHF.** RLHF is *all objective* (the reward model). Is there an RLHF-with-novelty story? Recent response-diversity preservation work resembles this idea but doesn't take it to the "ignore the reward" extreme.
- **Behaviour-space learning.** The paper hand-designs behaviour spaces. A natural follow-up is *learning* a behaviour descriptor (autoencoder, contrastive). Subsequent QD work (AURORA, etc.) does this; worth tracking.

## Connections

- **Foundational ancestor**: cited by [[papers/jiang-2022-rethinking-exploration]] as foundational for taming open-endedness.
- **Algorithmic-framework cousin**: [[papers/schmidhuber-2013-powerplay]] — POWERPLAY's "simplest unsolved problem" search is conceptually parallel to novelty search's "most novel candidate" selection. Same family of ideas, different vocabulary.
- **Position-paper context**: [[papers/clune-2019-ai-gas]] — Clune's AI-GAs framework explicitly cites novelty search and QD as foundational ingredients for pillar 3.
- **Modern intrinsic-reward analogue**: [[papers/guo-2022-byol-explore]] — BYOL-Explore's prediction-error-as-intrinsic-reward is the deep-RL descendant of novelty-as-signal.
- **Topic MOCs**: [[topics/open-ended-learning]], [[topics/exploration]], [[topics/evolutionary-computation]]
- **Author indices**: [[authors/joel-lehman]], [[authors/kenneth-stanley]]

## Selected quotes

> "Objective functions themselves may actively misdirect search toward dead ends."

> "Some problems are best solved by methods that ignore the objective."

(Both from the paper's central deception argument; specific page references require the journal PDF.)

## Source caveats

- Ingested manually via citation chase, not Readwise. Cite as Lehman & Stanley 2011, *Evolutionary Computation* 19(2):189–223, DOI 10.1162/evco_a_00025.
- Full paper PDF was fetched via the swarthmore.edu mirror; section/figure references are as named in that PDF. Specific equations, tables, and figure numbers should be re-verified against the published journal PDF before precise citation.
- Numerical results in the "Key results" section above are paraphrased from the paper; exact percentages, generation counts, and statistical comparisons should be re-fetched if citing in a paper.
