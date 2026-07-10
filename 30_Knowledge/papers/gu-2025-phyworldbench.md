---
type: paper
title: "PhyWorldBench: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models"
authors: ["Jing Gu", "Xian Liu", "Yu Zeng", "Ashwin Nagarajan", "Fangrui Zhu", "Daniel Hong", "Yue Fan", "Qianqi Yan", "Kaiwen Zhou", "Ming-Yu Liu", "Xin Eric Wang"]
year: 2025
venue: "ICLR 2026 (oral)"
url: https://arxiv.org/abs/2507.13428
rw_id: ""
topics: [world-models, video-generation]
priority: high
read_state: skimmed
relevance: ""
added: 2026-07-10
last_updated: 2026-07-10
---
<!-- ingested via /paper-search from arXiv (not via Readwise; no rw_id) -->

## TL;DR

PhyWorldBench is a large-scale benchmark that tests whether text-to-video (T2V) models actually obey physical laws, not just look photorealistic. It contains **1,050 curated prompts** covering **50 distinct physics phenomena** grouped into **10 main categories** across three tiers: Fundamental Physics, Composite Physics, and a novel **Anti-Physics** tier whose prompts deliberately violate real-world physics to test whether a model *understands* physics rather than merely memorizing training-data patterns. Evaluation uses a zero-shot MLLM-as-judge scheme with two binary criteria — Semantic Adherence (SA) and Physical Commonsense (PC) — sharpened by a **Context-Aware Prompt (CAP)** chain-of-thought that tells the judge the video is generated and makes it reason about physics before answering, lifting PC judging accuracy from ~61.6 to 75.1 ROC-AUC. Across **12 SOTA models** (5 proprietary, 7 open-source) and **12,600 generated videos**, physical realism is poor: the best model, **Pika 2.0, reaches only a 0.262 joint success rate** (both SA and PC satisfied), and performance monotonically collapses from Fundamental → Composite → Anti-Physics. A recurring failure is "rationalization" — models revert to plausible real-world physics or emit a near-static image rather than follow anti-physics instructions.

## Context & motivation

Text-to-video generation has advanced rapidly (Sora, Kling, Gen-3, Hunyuan, etc.), and such models are increasingly framed as "world simulators." But visual fidelity is not the same as physical correctness: a video can look sharp while objects float, pass through each other, or ignore gravity. The paper argues prior physics-in-video benchmarks are too small and too narrow to diagnose this. Table 1 positions PhyWorldBench against predecessors: **VideoPhy (688 prompts, 5 categories)**, **PhyGenBench (160 prompts, 27 categories)**, and **Physics-IQ (396 prompts, 5 categories)** — versus PhyWorldBench's **1,050 prompts and 50 physics categories**, a "significantly larger and more diverse testbed."

The paper's stated contributions are three:
1. Propose **PhyWorldBench**, a large-scale, multi-dimensional physics benchmark for evaluating the physics ability of video generation models.
2. Conduct an **extensive evaluation of twelve SOTA video generation models** (five proprietary, seven open-source) over **12,600 generated videos**, identifying key challenges in simulating real-world physics.
3. Study the **effect of prompt variation** on model performance and provide **prompt guidelines** for generating physics-following videos.

The distinctive conceptual move is the **Anti-Physics** tier: prompts that intentionally break physics ("wine with reversed gravity") to check whether models can follow such instructions while maintaining logical consistency — i.e., whether they *understand* physical laws rather than reproducing patterns from real-world training data.

## Method

### Problem formulation
Given a text prompt describing a physical scenario, a T2V model produces a short video. The benchmark asks two orthogonal questions about that output: (a) does it depict the requested content — the right objects and the main action/event (**Semantic Adherence, SA**); and (b) does the depicted motion/interaction obey the relevant physics — or, for anti-physics prompts, obey the *instructed* physics — (**Physical Commonsense, PC**). Both are scored as binary labels $\text{SA}, \text{PC} \in \{0,1\}$, and the headline metric is the fraction of videos with both jointly equal to 1.

### Core idea
Separate *what is shown* (semantics) from *whether it is physically right* (commonsense), test across a difficulty gradient that culminates in physics-violating prompts, and judge everything zero-shot with an MLLM whose reliability is boosted by a purpose-built chain-of-thought prompt.

### Benchmark construction
The prompt set is built combinatorially: **10 main categories × 5 subcategories × 7 scenarios × 3 prompt variations = 1,050 prompts**, spanning **50 distinct physics phenomena** (10 × 5 subcategories). The 10 main categories, organized into three tiers:

- **Fundamental Physics (6 categories)** — basic laws such as object motion, energy transfer, and optics. Categories include: Object Motion & Kinematics, Interaction Dynamics, Energy Conservation, Fluid & Particle Dynamics, Lighting & Shadows, Deformations & Elasticity.
- **Composite Physics (3 categories)** — real-world phenomena emerging from multiple interacting principles: Scale & Proportions, Rigid Body Dynamics, Human & Animal Motion.
- **Anti-Physics (1 category)** — scenarios intentionally designed to violate real-world physics.

Each scenario is expressed at three prompt granularities (the "prompt variations"):
- **Event Prompt** — a concise, minimal description of an event; tests generation from minimal input.
- **Physics-Enhanced Prompt** — the Event Prompt enriched naturally with the relevant physics phenomenon.
- **Detailed Narrative Prompt** — enriched with vivid detail and context for more immersive, visually rich output.

### Evaluation methodology
Judging is **zero-shot MLLM-as-judge**. Several MLLMs are tested as judges — **GPT-o1, GPT-4o, Gemini-2.0-flash, and Qwen-VL-2.0**. For each video the judge assigns two binary labels:

- **Semantic Adherence** $\text{SA} \in \{0,1\}$: is the correct number of objects present and is the main action/event depicted?
- **Physical Commonsense** $\text{PC} \in \{0,1\}$: does the video adhere to real-world physics (per scenario "Key Standards"), or, for anti-physics prompts, the instructed physics?

The reported scores are the fractions of videos with $\text{SA}=1$, with $\text{PC}=1$, and with both jointly satisfied (the primary success metric).

The key methodological contribution is the **Context-Aware Prompt (CAP)** — a two-stage chain-of-thought for the judge:
1. **Description stage** — the MLLM produces a thorough description of the video, covering Object, Event, and Observations.
2. **Reasoning stage** — the MLLM reasons through potential physics issues before committing to a final Yes/No.

Crucially, CAP **explicitly tells the MLLM that the video is generated rather than real**, which prevents the judge from charitably assuming physical correctness. This raises Physical Commonsense judging accuracy from **~61.6 to 75.1 ROC-AUC**. Ablation (Table 2): CAP reaches **80.3 SA / 75.1 PC**, versus vanilla GPT-o1 at **75.4 SA / 61.6 PC** — an absolute **+13.5** on PC. Removing the CoT drops PC to 73.6; removing the "this is generated" context drops PC to 65.6 — so most of the gain comes from telling the judge the video is synthetic.

**Human study protocol.** Ground truth for the judge validation comes from **Amazon Mechanical Turk** workers paid **$18/hour**. Each video is labeled by **three independent annotators** with the final label by **majority vote**, across three binary criteria: object existence, event presence, and visibility of the key physics phenomenon. The human study spans the full **12,600 videos** (1,050 prompts × 12 models). No single inter-rater reliability coefficient is reported.

## Experimental setup

**Models evaluated (12 total):**
- *Proprietary (5):* Sora-Turbo (OpenAI), Gen-3 (Runway), Kling 1.6, Pika 2.0, Luma.
- *Open-source (7):* Hunyuan 720p (Tencent), Open-Sora 2.0, Open-Sora-Plan 1.3, CogVideoX-1.5, Step-video-T2V, Wanx-2.1 (Wan), LTX-Video.

**Scale:** 1,050 prompts × 12 models = **12,600 generated videos**.

**Metrics:** fraction with SA=1, fraction with PC=1, and joint (SA=1 and PC=1) success rate — the headline number. Judge quality itself is validated against human majority-vote labels via ROC-AUC.

## Key results

*Summarized — see paper Table 3/rankings for full detail.*

**Overall joint success (both SA and PC):** physical realism is uniformly weak. Best-to-worst:
- Pika 2.0 — **0.262** (best overall)
- Sora-Turbo — 0.208
- Wanx-2.1 — 0.189 (best open-source)
- Kling-1.6 — 0.188
- Hunyuan 720p — 0.185
- Luma — 0.184
- Step-video-T2V — 0.173
- Open-Sora 2.0 — 0.170
- CogVideoX-1.5 — 0.163
- Gen-3 — 0.130
- Open-Sora-Plan 1.3 — 0.075
- LTX-Video — 0.062

Even the best model succeeds only ~26% of the time; proprietary models lead on average but the best open-source model (Wanx-2.1) edges most proprietary ones.

**Difficulty gradient.** Performance declines monotonically across tiers. Approximate semantic-adherence bands: Fundamental ~24–38%, Composite ~16–34%, Anti-Physics ~6–23%. Multi-object / multi-force scenes are especially hard — e.g., for a feather-and-stone differential-fall scenario, "all models fail to present the phenomenon correctly."

**Anti-Physics behavior.** Models overwhelmingly **revert to real-world physics rather than following the violating instruction**. Failures "often stem from models interpreting prompts in a more 'reasonable' way rather than attempting and failing." Example: prompted with "the wine in the glass has reversed gravity," a model generates a near-still image consistent with normal gravity. This "static-scene rationalization" was Pika's failure mode in ~34% of its failures, versus ~4–6% for weaker models — i.e., the strongest model fails by being too conservative.

**Prompt sensitivity.** Softening abrupt/erratic changes helps: ~22.1% success for abrupt changes vs ~42.1% for softened prompts. Simpler scenes beat complex ones broadly (~18.3% vs ~11.7%; Pika ~40% simple vs ~15% complex).

**Judge validation.** CAP-based MLLM judging aligns well with humans (80.3 SA / 75.1 PC ROC-AUC) and is near-neutral in effect on aggregate scores (≈0.001 change in SA, ≈0.014 in PC when applied).

## Ablations

The CAP ablation (Table 2) is the load-bearing one: it isolates *why* the judge works. Vanilla GPT-o1 gives 61.6 PC; adding CoT reasoning alone → 73.6; adding the "video is generated" context is what pushes PC to 75.1 (full CAP → 80.3 SA / 75.1 PC). So the dominant lever is disabling the MLLM's default assumption that videos depict real, physically-valid scenes. Prompt-variation analysis (event vs physics-enhanced vs detailed narrative) functions as a second ablation, showing softened/simpler prompts materially raise success and yielding the paper's prompt guidelines.

## Limitations

- The benchmark depends on **closed-source MLLM judges** (GPT-o1/GPT-4o/Gemini) for PC scoring; judge accuracy on PC caps at ~75 ROC-AUC, so ~1 in 4 physics judgments may be wrong, and reproducibility is tied to proprietary models that drift over time. _(This is the exact critique later benchmarks raise — see Connections.)_
- Success is reduced to **binary SA/PC labels**; graded or continuous physical error is not measured.
- No single inter-rater reliability coefficient is reported for the human study, making annotation noise hard to quantify. _(analyst-flagged)_
- Short generated clips only; long-horizon physical consistency is out of scope. _(analyst-flagged)_

## Why it matters [analyst's view]

PhyWorldBench crystallizes a claim the vault keeps circling: current T2V "world models" are photorealistic surface fitters, not physics engines — the best system clears both semantics and physics on barely a quarter of prompts. The **Anti-Physics** tier is the sharpest instrument here: by showing models default to "reasonable" physics even when told not to, it operationally separates *pattern reproduction* from *physical understanding*, which is exactly the distinction [[papers/zhao-2026-phyworld]] probes from the scaling side and [[papers/esmati-2026-invisible-hand-physics]] probes mechanistically. The rationalization finding — the strongest model failing by being *too conservative* (static images) — is a nice counter to the naive "just scale it" narrative. The obvious tension is methodological: PhyWorldBench leans entirely on closed-source MLLM judges, and [[papers/lin-2026-phyground]] (PhyGround) explicitly critiques exactly this, arguing for grounded/verifiable physical evaluation instead of a black-box LLM verdict. Read the two together to see the benchmark-design debate playing out. For anyone building physics-aware generation or alignment ([[papers/yuan-2026-physics-alignment]]), the tiered prompt taxonomy and CAP judging recipe are reusable evaluation infrastructure.

## Open questions / things to verify

- How stable are the rankings under a different (open-source) judge? The paper tests several MLLMs but standardizes on GPT-o1+CAP — does Pika still lead with a Qwen-VL judge?
- The ~75 PC ROC-AUC ceiling means the joint success numbers carry non-trivial judge error; how sensitive are the 0.262 vs 0.208 gaps to that noise?
- Exact per-category breakdowns (the 24–38% / 16–34% / 6–23% bands are approximate from the fetched text) — verify against the paper's tables before citing precisely.
- Does the anti-physics "rationalization" persist if the violation is stated more forcefully or repeated? The static-image failure may be a prompt-strength artifact.

## Connections

- Contrasts with / debated by: [[papers/lin-2026-phyground]] — PhyGround, a direct competitor benchmark that critiques PhyWorldBench's reliance on closed-source MLLM judges and argues for grounded physical evaluation.
- Related (physics in video/world models): [[papers/zhao-2026-phyworld]], [[papers/esmati-2026-invisible-hand-physics]], [[papers/yuan-2026-physics-alignment]], [[papers/joseph-2026-physics-video-world-models]], [[papers/ye-2026-world-action-models]]
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]]
- Author indices: [[authors/jing-gu]], [[authors/xin-eric-wang]], [[authors/ming-yu-liu]]

## Selected quotes

> "Prompts intentionally violate real-world physics, enabling the assessment of whether models can follow such instructions while maintaining logical consistency." — Anti-Physics definition

> "failures often stem from models interpreting prompts in a more 'reasonable' way rather than attempting and failing." — on anti-physics behavior
