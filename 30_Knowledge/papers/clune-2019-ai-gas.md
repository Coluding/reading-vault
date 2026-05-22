---
type: paper
title: "AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence"
authors: ["Jeff Clune"]
year: 2019
venue: arXiv
url: https://arxiv.org/abs/1905.10985
rw_id: "manual:clune-2019"
topics: [open-ended-learning, exploration, general-intelligence, meta-learning, position-paper]
priority: high
read_state: skimmed
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# AI-GAs: AI-generating algorithms

**TL;DR** — Clune's 2019 position paper proposes **AI-Generating Algorithms (AI-GAs)** as an alternative to the dominant *manual AI* paradigm (hand-engineer pieces of intelligence; figure out how to combine them later). AI-GAs would automatically discover the components of general AI through three jointly-pursued pillars: (1) **meta-learning architectures**, (2) **meta-learning the learning algorithms themselves**, and (3) **generating effective learning environments**. The deep argument is a historical induction: ML has repeatedly replaced hand-engineering at each layer of the stack with learned counterparts (hand-coded vision → hand-designed features like HOG/SIFT → end-to-end learned vision); AI-GAs are the logical next move — replace hand-engineering of the *whole* stack of architecture, learner, and training distribution with end-to-end learning. The third pillar (learning environments) is named as the least-studied, hardest, and most likely to host the next decade's breakthroughs.

> _Ingested manually as a citation chase from [[papers/jiang-2022-rethinking-exploration]] (cited as foundational for open-ended self-improving systems). Source: full paper via ar5iv HTML render; this revision reflects a full-paper read, not an abstract-only triage._

## Context & motivation

Clune frames general AI as the "most ambitious scientific quest in human history" and identifies the dominant paradigm as the **manual AI approach**:

> "The dominant approach in the machine learning community is to attempt to discover each of the pieces required for intelligence, with the implicit assumption that some future group will complete the Herculean task of figuring out how to combine all of those pieces into a complex thinking machine. I call this the 'manual AI approach'."

The structure of the critique:

- **Two-phase contradiction.** The manual path tacitly assumes Phase One (discover individual building blocks) is the hard part and Phase Two (combine 60+ building blocks into a single integrated thinking machine) is "left as an exercise." Clune lists 60+ candidate building blocks in Table 1 (attention, memory, planning, value learning, intrinsic motivation, modularity, hierarchy, abstraction, transfer, etc.) and points to combinatorial explosion: with that many pieces and non-linear interactions, the combination problem is genuinely intractable.
- **Cultural / institutional mismatch.** Current academic incentives reward small teams producing a few papers per year; the integration problem realistically needs "an Apollo program" — a large, dedicated, long-horizon effort.
- **Incomplete knowledge.** It's unclear how many essential building blocks remain undiscovered, or which of the known building blocks are *the right variant*.

Clune doesn't dismiss the manual path — he explicitly notes it "has a chance of being the fastest path to AI" — but argues it's not the *only* path, and that AI-GAs deserve substantial investment in parallel.

## The historical lineage argument

The paper's most compelling rhetorical move is a historical induction. The argument: ML has repeatedly replaced hand-engineering with learning, and AI-GAs continue that trajectory.

> "When we want to create an intelligent system, we as a community often first try to hand-design it... Once we realize that is too hard, we then try to hand-code some components... Ultimately, we realize that with sufficient data and computation, we can learn the entire system."

The flagship case study: **computer vision**.

- Pre-2000s: hand-coded vision systems → failed at scale.
- 2000s: hand-designed features (HOG, SIFT) + learned classifiers → worked reasonably.
- 2012+: end-to-end learned vision systems (AlexNet onward) → blew past the hand-feature regime.

Same pattern named for sound, touch, NLP, machine translation. Same pattern more recently for *architecture* itself: a NAS-discovered architecture beat the over-optimized hand-designed champions on CIFAR.

By induction: the *next* layer of hand-engineering to fall is the entire stack (architecture, learner, training environment) at once.

## The three pillars

### Pillar 1 — Meta-learning architectures

Replace hand-designed neural architectures with automated architecture search.

- **Status in 2019:** Architecture search had just begun producing state-of-the-art results on CIFAR and competitive results on ImageNet. The pillar is the most mature of the three.
- **Open research problems Clune names:** encodings (representations) for architectures, search operators, and the production of architectures that are *regular, modular, and hierarchical*.
- **Cited named architectures whose hand-design pillar 1 would replace:** convolution, LSTMs, highway networks, residual networks, attention mechanisms.

### Pillar 2 — Meta-learning the learning algorithms

Replace fixed learning algorithms (SGD, Adam, fixed weight-update rules) with learned learning algorithms.

- **Two families named.**
  1. **Gradient-based meta-learning (MAML family).** Differentiate through the learning process to find initial weights, learning rates, or update rules that produce fast adaptation.
  2. **RNN-based meta-learning.** A recurrent network "implements" a learning algorithm via its activations — the recurrent dynamics *are* the learning algorithm. No explicit weight update.
- **Clune's own group's contributions named:**
  - **Differentiable plasticity** — networks whose weights change *intralife* via Hebbian-style updates, trained end-to-end. Reported to beat LSTMs on memory-intensive tasks.
  - **Neuromodulation** — neurons that gate the learning rates of other connections. Reported on 600 sequential tasks (over 9000 SGD updates) as an approach to catastrophic forgetting.
- **Acknowledged bottleneck:** "Meta-learning is currently computationally expensive… limiting the complexity of tasks that we can train on." Pillar 2 is the most compute-hungry per result.

### Pillar 3 — Generating effective learning environments

The pillar Clune flags as "the least-studied, least-understood, and likely hardest" — and the one he expects will host the most discoveries in the coming decade.

**Two sub-approaches:**

1. **Target-task environment generation.** Generate training data / environments tuned to a *specific* downstream goal. Flagship example: **Generative Teaching Networks (GTNs)** — deep nets that synthesize training data (typically unrecognizable, "alien" inputs) which nevertheless train student networks ~4× faster than real MNIST data.
2. **Open-ended environment generation.** Generate an *expanding set* of environments without a predetermined target — analogous to natural evolution producing the Cambrian explosion from simple early environments.

**Key mechanisms named for the open-ended sub-approach:**

- **Behavioural diversity / novelty search** (Lehman & Stanley) — see [[papers/lehman-2011-novelty-search]]. Reward novelty, not objective performance.
- **Quality-Diversity (QD) algorithms** — search for *the highest-performing solution in each region* of a behaviour space. Genealogy: Novelty Search with Local Competition → **MAP-Elites** (Nature paper, robot damage recovery in 1–2 minutes). Clune frames QD as a foundational primitive.
- **Paired Open-Ended Trailblazer (POET)** — explicitly named. POET co-evolves agents and environmental challenges, with **goal-switching**: agents from one environment can transfer to another niche and become the new champion there. Clune sketches a vision of POET-like systems producing "water worlds, desert worlds, and mountain worlds" with adaptive radiation between niches.
- **Innovation Engines** — goal-switching applied to image generation, producing diverse high-quality images.
- **Go-Explore** — descendant of MAP-Elites; achieved SOTA on Montezuma's Revenge and Pitfall.

**The hardest unsolved problem inside pillar 3.** Clune explicitly admits he does not have an answer:

> "What is the reward function (aka loss function or fitness function) for the environment generator?"

He sketches preliminary candidates — transfer learning quality, learning progress, environment difficulty curves — but flags this as the open problem inside the open problem.

## "Why investigate now?"

Five reasons named for prioritising AI-GAs in 2019+:

1. **Scientific interest.** Studying the conditions under which a simple outer-loop algorithm produces general intelligence is exactly the question evolution answered on Earth — Earth's evolution being "the first general-intelligence-generating algorithm."
2. **Scalability with compute.** "History has shown that the algorithms that tend to win over the long haul are simple ones that can take advantage of massive amounts of computing." AI-GAs are compute-hungry by design and absorb compute easily as it becomes available — explicit Sutton's-bitter-lesson alignment.
3. **Lower combinatorial burden.** Far fewer researchers pursue AI-GAs than manual AI; the search space is "more manageable" with fewer building blocks per pillar.
4. **Economic value pre-AGI.** Even before general AI, automating architecture search, optimizer design, and environment generation produces immediate downstream value.
5. **Diversity of resulting AIs.** AI-GAs would produce "a much wider diversity of intelligent beings" than the manual path, which is limited by human imagination.

## Limitations & counter-arguments Clune acknowledges

- **Compute may run out before AI-GAs catch up.** The manual path is currently more compute-efficient. "Whether we will have sufficient compute to see this pillar succeed before general AI is produced by the manual path is an open question."
- **Near-/medium-term advantage to the manual path.** Clune predicts the manual path will continue to produce higher-performing systems in the near term, like HOG/SIFT did pre-2012. The crossover where end-to-end learned systems pass hand-designed ones may come later.
- **Environment-generator reward function unresolved** — see above.
- **Deception and sparse reward problems.** Direct optimisation for "general intelligence" hits the same deception that direct optimisation for any vague target hits (IQ, carbon emissions, etc., used as analogies).
- **Grounding to human-relevant problems.** Open-ended environment generation could produce environments that train powerful but useless or dangerous agents. "How do we constrain the generation of environments to be those we find interesting and/or that produce intelligence that helps us solve real-world problems?" — open.
- **AI-GAs ≠ "evolutionary approach."** Clune explicitly warns against this framing — the outer-loop optimiser could be evolutionary *or* gradient-based *or* RL-based.

## Alignment / safety

Clune addresses safety briefly (Section 4) but defers detailed discussion. The honest summary:

- AI-GAs would produce "a complex machine whose inner workings we do not understand" — interpretability is harder, not easier, under AI-GAs.
- However, AI-GAs could also produce *many* general AIs, which might help us identify what is necessary vs. incidental in intelligent systems.
- The paper does not deeply engage modern alignment risks (this is 2019, pre-RLHF-mainstream).

[analyst's view] This is the weakest section of the paper. From 2026, you can read pillar 3 as an *unaligned* superintelligence playbook unless safety is welded into the environment-generator reward function. The fact that Clune explicitly does not have a candidate reward function for environment generation makes this concerning — pillar 3 essentially asks "let's automate the data/environment side of training without yet knowing how to specify what we want."

## Predictions for the next decade

Clune ranks the AI-GA path as more likely than the manual path to produce general AI first, "but with high uncertainty." Specific predictions:

1. The trend of hand-design → learned design will continue across modalities.
2. End-to-end learned systems will eventually surpass and then "rapidly far exceed" manual ones — same pattern as 2012 vision but at a higher level of the stack.
3. Pillar 3 will host most of the upcoming history-making discoveries.
4. The eventual successful system will look more like an AI-GA than a manual integration, "if everything is being learned from a few basic principles."
5. Both paths should be pursued in parallel; the question is funding ratio, not exclusivity.

## Why it matters [analyst's view]

Three reasons this paper is worth holding in the vault, updated after the full read:

1. **It names the next layer of automation, sharply.** "Manual AI approach" is rhetorically clean and turns out to be predictive. By 2026, RLHF, synthetic data generation, autocurricula, self-play, and LLM-self-improvement loops are all instances of AI-GAs' third pillar in practice — even if the practitioners running them don't use Clune's vocabulary.

2. **It is the cleanest pre-LLM articulation of the "data and environment are the bottleneck" thesis.** Three years before [[papers/jiang-2022-rethinking-exploration]] formalised the outer/inner-loop distinction, Clune was already arguing data/environment generation deserves equal investment to architecture and optimisation. Jiang 2022 is partly a refinement of this thesis with sharper formalism; the conceptual debt is real.

3. **The QD genealogy in Section 3 is the single best map.** Novelty Search ([[papers/lehman-2011-novelty-search]]) → Novelty Search with Local Competition → MAP-Elites → POET → Go-Explore → Innovation Engines, all in one place, with the conceptual moves explicit. If you only ever read one Clune-group paper, this is a strong candidate because of the genealogy alone.

Two things this paper *underweights* from a 2026 perspective:

- **Specification problem in pillar 3.** "What's the reward function for environment generation?" is admitted as unsolved but treated as a research problem rather than an existence-of-aligned-solution problem. Three years later, alignment researchers would call this much harder than Clune frames it.
- **The "manual path" is closer to AI-GAs than 2019-Clune frames it.** Modern frontier-lab pipelines blend pillar 3 (synthetic data, evals, RLHF) with manual architecture choices. The binary is fuzzier in practice.

## Open questions

[analyst's view]
- **What does pillar 2 actually look like in practice?** Meta-learning learning algorithms has fewer flagship 2026 results than pillars 1 (NAS) and 3 (autocurricula). Why?
- **Cross-link to [[papers/baek-2026-gram]] and [[papers/wang-2025-hierarchical-reasoning-model]]:** these papers *learn the latent dynamics of reasoning* — could that be a version of pillar 2 for reasoning tasks?
- **Pillar 3 in LLM post-training:** RLHF and synthetic data are pillar-3 in practice. But these typically don't *also* run pillar 1 and pillar 2 jointly. Are the pillars actually compounding, or has the field tractably separated them?
- **POET in the modern era.** What is POET's 2026 descendant? Open-ended LLM red-teaming? RL-from-LLM-judge synthetic curricula? Worth mapping.

## Connections

- **Conceptual successor / refinement**: [[papers/jiang-2022-rethinking-exploration]] — Jiang et al. cite Clune's AI-GAs explicitly and build the outer/inner-loop formalism on top of this thesis.
- **Pillar 3 ancestor**: [[papers/lehman-2011-novelty-search]] — novelty search is the empirical demonstration that objective optimisation can hurt; QD and POET build on it.
- **Pillar 3 algorithmic cousin**: [[papers/schmidhuber-2013-powerplay]] — POWERPLAY's "search the joint (task, solver-update) space" is essentially pillar 3 stated in different vocabulary.
- **Inner-loop counterpart**: [[papers/guo-2022-byol-explore]] — BYOL-Explore is a concrete instantiation of intrinsic-reward exploration that fits inside AI-GAs' broader vision.
- **Topic MOCs**: [[topics/open-ended-learning]], [[topics/exploration]]
- **Author indices**: [[authors/jeff-clune]]

## Selected quotes

> "Perhaps the most ambitious scientific quest in human history is the creation of general artificial intelligence, which roughly means AI that is as smart or smarter than humans."

> "The dominant approach in the machine learning community is to attempt to discover each of the pieces required for intelligence, with the implicit assumption that some future group will complete the Herculean task of figuring out how to combine all of those pieces into a complex thinking machine. I call this the 'manual AI approach'. This paper describes another exciting path that ultimately may be more successful at producing general AI."

> "When we want to create an intelligent system, we as a community often first try to hand-design it… Once we realize that is too hard, we then try to hand-code some components… Ultimately, we realize that with sufficient data and computation, we can learn the entire system."

> "History has shown that the algorithms that tend to win over the long haul are simple ones that can take advantage of massive amounts of computing."

> "Generative Teaching Networks (GTNs) … generate data [that] look completely unrecognizable and alien, yet still the student network learns… four times faster than when training on real MNIST training data."

> "What is the reward function (aka loss function or fitness function) for the environment generator?" *[Clune: "I do not have an answer to what this environment-generator reward function should be."]*

## Source caveats

- Ingested manually via citation chase, not Readwise. No Readwise rw_id; cite as Clune 2019, arXiv:1905.10985.
- Content extracted from the ar5iv HTML render of the full paper, not just the abstract. Specific equations / page numbers not verified against the PDF — re-check before citing exact table or equation references.
- The PDF render of the figures was not parsed; any claim about figure content is from the surrounding prose, not the figures themselves.
