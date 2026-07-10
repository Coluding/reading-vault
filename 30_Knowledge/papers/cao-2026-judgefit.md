---
type: paper
title: "Each Judge Its Own Yardstick: Discovering Per-VLM Taxonomies for Physical Video Evaluation"
authors: ["Yu Cao", "Ziquan Liu", "Zhensong Zhang", "Jiankang Deng", "Shaogang Gong", "Jifei Song"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2606.22918
rw_id: ""
topics: [video-generation, world-models]
priority: medium
read_state: skimmed
relevance: ""
added: 2026-07-10
last_updated: 2026-07-10
---
<!-- ingested via /paper-search from arXiv (not via Readwise; no rw_id) -->

## TL;DR

Video generators and world models increasingly rely on vision-language models (VLMs) as automated judges of physical consistency — providing reward signals, ranking decisions, and data-filtering criteria — but different VLMs encode physics through distinct internal representations, so a single global evaluation rubric hands every VLM the same axes of competence regardless of what it can actually perceive. **JudgeFit** discovers a *per-VLM* evaluation taxonomy: it prompts the target VLM to enumerate physics errors on a small video set, clusters those into a seed taxonomy, then iteratively calibrates each dimension's scores against human physical-commonsense ratings, diagnoses dimensions that are uninformative / unstable / redundant, and asks an LLM to repair them until convergence. Instantiated as a benchmark over **16 VLMs spanning eight model families**, the refined per-VLM taxonomy beats the global-schema baseline on held-out videos **for every VLM tested, with a mean relative improvement of ~32%** (Spearman ρ rising from 0.239 baseline to 0.315 refined). Beyond aggregate accuracy, the per-VLM profiles expose model-specific blind spots that overall rankings cannot anticipate, and reliability patterns differ markedly across families. The core message: the yardstick should be fit to the judge, not the judge to the yardstick.

## Context & motivation

Automated physical-consistency evaluation has become load-bearing for the video-generation / world-model stack: VLM judges supply reward signals for RL, ranking decisions for model selection, and filtering criteria for training data. The problem the paper attacks is that VLMs "differ substantially in training data and architecture, encoding physical phenomena through distinct internal representations." Prior practice applies one global evaluation schema — the same fixed set of physics dimensions — to every judge. The authors argue this is a category error: "A single global evaluation schema therefore gives every VLM the same axes of competence, regardless of what each can actually perceive." A dimension a given VLM cannot reliably perceive contributes noise, while an axis it *is* sensitive to may be missing from the global rubric entirely. The stated contribution is JudgeFit, an iterative refinement procedure that discovers a taxonomy tailored to each VLM, plus its instantiation as a benchmark over 16 VLMs. This work sits directly in the emerging "who judges the video judge" line — related in the vault to PhyGround's specialized open judge and PhyWorldBench's fine-tuned MLLM judge (see Connections).

## Method

### Problem formulation

Given a VLM judge $v$, an annotation set $\mathcal{D}_{annot}$ with per-video human physical-commonsense ratings $y(x) \in \{1,\dots,5\}$, and a disjoint test set $\mathcal{D}_{test}$, JudgeFit outputs a **per-VLM taxonomy** $\mathcal{T} = \{(d_1,\delta_1),\dots,(d_k,\delta_k)\}$, where each dimension $d_i$ carries a one-line definition $\delta_i$ grounded in an observable physics violation. The VLM scores each video along the taxonomy to produce per-dimension scores $\mathbf{s}^v_{\mathcal{T}}(x) \in [0,5]^k$ (error severity per dimension). The objective is to find a taxonomy whose scores maximize agreement with the human ratings $y$.

### Core idea

Rather than imposing one global rubric on all judges, give **each judge its own yardstick**: build the taxonomy only from errors that judge $v$ actually reports, then keep only the dimensions on which $v$ scores in a way that tracks human judgement. Because the seed taxonomy $\mathcal{T}_0$ is "built only from errors the judge reports on its own, its dimensions are exactly those the judge already attends to."

### JudgeFit algorithm

**Stage 1 — Seed (initial taxonomy construction).**
1. *Open annotation:* the VLM $v$ lists physics errors in free-form text for each video in $\mathcal{D}_{annot}$, stratified across the quality spectrum.
2. *Clustering:* an LLM pools all descriptions and clusters them into named, defined categories, forming the seed taxonomy $\mathcal{T}_0$ (typically ~4 dimensions, range 3–7).

**Stage 2 — Refine (iterative diagnosis-guided editing).** Each round:
1. *Scoring:* the VLM re-scores every video on the current dimensions; each dimension is scored 3 times and averaged, on scale $[0,5]$.
2. *Calibration:* fit a **ridge regression** ($\lambda = 1.0$) mapping the per-dimension scores $\mathbf{s}(x)$ to the human rating $y(x)$ via $f_{\mathcal{T}}$, using **5-fold cross-validation** to avoid leakage, producing calibrated predictions $\hat{y}_{full} = f_{\mathcal{T}}(\mathbf{s})$.
3. *Diagnosis:* for each dimension $i$ compute three signals — the standardized regression coefficient $\beta_i$; the sign consistency across folds $\pi_i \in [0,1]$; and the Spearman rank correlation with human ratings,
$$\rho_i = \frac{\mathrm{cov}(\mathrm{rk}(s_i),\,\mathrm{rk}(y))}{\sigma_{\mathrm{rk}(s_i)}\cdot\sigma_{\mathrm{rk}(y)}} \quad\text{(Eq. 1)}$$
where $\mathrm{rk}(\cdot)$ is the rank transform over videos, $s_i$ the dimension-$i$ scores, and $y$ the human ratings. Also compute pairwise dimension correlations $r_{ij}$.
4. *Fault detection (Eq. 2):*
$$|\rho_i| < \tau \Rightarrow \text{drop (uninformative)}$$
$$\pi_i < \kappa \Rightarrow \text{redefine (unstable)}$$
$$\max_{j\neq i} r_{ij} > \gamma \Rightarrow \text{merge (redundant)}$$
with thresholds $\tau = 0.10$, $\kappa = 0.60$, $\gamma = 0.60$. *Coverage faults* are found separately: residuals between $\hat{y}_{full}(x)$ and $y(x)$ reveal error classes no current dimension captures.
5. *LLM-driven repair:* an LLM editor receives the diagnostic report and proposes $m = 2$ candidate edits, drawn from five operations: `merge(dim_a, dim_b)`, `split(dim)`, `redefine(dim)`, `add(name, definition)`, `drop(dim)`.
6. *Evaluation & selection:* for each candidate taxonomy $\mathcal{T}_j$ the VLM re-scores all videos and the selection objective is computed,
$$J(\mathcal{T}) = \underbrace{\rho(\hat{y}_{full})}_{\text{ranking quality}} + \underbrace{[\rho(\hat{y}_{full}) - \rho(\hat{y}_{halo})]}_{\text{specificity gain}} \quad\text{(Eq. 3)}$$
where $\hat{y}_{halo}$ is a "halo" baseline prediction using only the *mean* dimension score (i.e. no per-dimension structure). The second term rewards taxonomies whose dimensions add discriminative signal beyond a single averaged score. A candidate is accepted only if $J_j > J^*$ (the current best).
7. *Convergence:* iterate until no edit improves $J$, or the budget $T_{max} = 3$ rounds is exhausted. Post-refinement taxonomies typically hold 3–8 dimensions.

### Training procedure

There is no model training — JudgeFit is a prompting-plus-calibration pipeline; the only fitted object is the per-round ridge regression ($\lambda=1.0$, 5-fold CV). The seed and refine stages call the target VLM (scoring, 3× per dimension) and an LLM (clustering, repair).

### Inference / sampling

_not addressed by the source_ (JudgeFit evaluates generated videos rather than generating them; at deployment the discovered taxonomy is simply applied to score new videos with the target VLM).

## Experimental setup

- **Judges:** 16 VLMs across eight model families. Reported models include Gemini-3-Flash, Gemini-3-Flash-Thinking, Gemma-4-31B (Google); Seed-2.0-Lite, Seed-2.0-Lite-Thinking (ByteDance); MiMo-v2.5, MiMo-v2.5-Thinking, MiMo-v2-Omni, MiMo-v2-Omni-Thinking (Xiaomi); Qwen2.5-VL-7B, Qwen3-VL-8B-Instruct (Alibaba); GLM-4.6V, GLM-4.6V-Thinking, GLM-5V-Turbo (Zhipu); Nova-2-Lite (Amazon); Reka-Edge (Reka).
- **Calibration/eval data:** the **VideoPhy-2** benchmark with per-video human ratings and physical-rule annotations — 300 videos total, split into a 200-video annotation set and a disjoint 100-video test set, balanced across the five rating levels (1–5). (Annotator count not specified.)
- **Metric:** Spearman rank correlation $\rho$ between per-dimension VLM scores and human ratings.
- **Baseline:** a global-schema baseline (referred to as the PhyGen / global rubric baseline).

## Key results

- **Headline:** the refined taxonomy beats the global-schema baseline on held-out videos **for every VLM tested**, mean relative improvement **~32%** — Spearman ρ from **0.239 (baseline) → 0.315 (refine)**.
- **Best / worst judge:** strongest is Gemini-3-Flash-Thinking at **ρ = 0.484**; weakest is GLM-4.6V at **ρ = 0.116**.
- **Per-family reliability differs markedly:** Gemini family shows consistent gains (**+0.126 to +0.138** over baseline); GLM-4.6V-Thinking **+0.126**; the MiMo family gains least (three variants under **+0.06**); Reka-Edge starts from the weakest seed but recovers to competitive level (**+0.081**).
- **Interpretive value:** per-VLM profiles expose model-specific blind spots that aggregate rankings cannot anticipate.

## Ablations

- **Seed → Refine progression (Fig. 3):** most gain arrives in round 1 — Seed→Round 1 **+0.07** Spearman ρ, Round 1→2 **+0.02**, Round 2→3 **+0.02**; "most models plateau by round 2."
- **Thinking-mode effect (Table 2):** four of five families improve with their "thinking" variants; one (MiMo-v2-Omni) declines. The effect does not correlate with base capability.
- **Convergence:** refinement stops when no edit improves the selection objective $J$.

## Limitations

Authors' own (labelled as such):
1. JudgeFit "requires human-annotated data carrying both ratings and per-video problem annotations," though they note it "needs far fewer such annotations than existing approaches."
2. "The seed and refine stages call an LLM, whose reproducibility is not guaranteed... but its randomness does not affect deployment: once a better taxonomy is found it can be applied directly." Validation "across a broader range of settings" is left to future work.

Honest-reader flags [analyst's view]: evaluation rests on a single human-rating source (VideoPhy-2, 300 videos, 100 held out), so the ~32% figure is a within-benchmark result; the absolute ceiling is low (best ρ ≈ 0.484), meaning even the best fitted VLM judge only moderately correlates with humans; and the family list as extracted names seven vendors, so the "eight families" grouping should be confirmed against the paper's own table.

## Why it matters [analyst's view]

This reframes VLM-as-judge for physics from "pick the best judge + rubric" to "fit the rubric to the judge." That is a meaningful shift for anyone using VLM reward signals in a video-generation or world-model RL loop: a mis-specified global rubric silently injects noise from dimensions the judge cannot perceive, and JudgeFit's diagnosis-and-prune loop is a concrete way to strip that noise. It complements — rather than competes with — the "build one strong specialized judge" line: PhyGround's PhyJudge-9B and PhyWorldBench's fine-tuned MLLM judge invest in a single better judge, whereas JudgeFit accepts heterogeneous judges and adapts the measurement to each. The low absolute ceiling (ρ ≈ 0.48 best) is itself a useful signal that VLM physics-judging remains far from human-level, which cross-validates the skepticism about automated physical-consistency metrics running through PhyWorldBench and the invisible-hand-physics work.

## Open questions / things to verify

- Confirm the exact "eight model families" partition against the paper's table (the extracted vendor list reads as seven).
- Does the discovered taxonomy transfer across *datasets* (only VideoPhy-2 is used) or across video-generator distributions?
- How stable is a discovered taxonomy under the acknowledged LLM stochasticity — do repeated JudgeFit runs on the same VLM converge to similar dimensions?
- The specificity-gain term relies on a "halo" mean-score baseline; how sensitive are results to that choice of $\hat{y}_{halo}$?
- Does a JudgeFit-fitted judge yield better *downstream* reward signals (e.g. RL training outcomes), not just higher held-out ρ?

## Connections

- Related (VLM-judge / physical-video evaluation): [[papers/lin-2026-phyground]] — builds the open specialized judge PhyJudge-9B and measures judge bias; strongest link.
- Related: [[papers/zhao-2026-phyworld]] — PhyWorldBench also fine-tunes a video-language judge for physical consistency.
- Related: [[papers/esmati-2026-invisible-hand-physics]] — physical-plausibility evaluation of video/world models.
- Related: [[papers/joseph-2026-physics-video-world-models]] — physics in video world models.
- Topic MOCs: [[topics/video-generation]], [[topics/world-models]]
- Author indices: _not created (per ingest instructions)_
