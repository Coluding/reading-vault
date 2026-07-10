---
type: paper
title: "CRONOS: Benchmarking Counterfactual Physical Consistency in Video Models"
authors: ["León Begiristain", "Olaf Dünkel", "Adam Kortylewski"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2605.23699
rw_id: ""
topics: [world-models, video-generation]
priority: medium
read_state: skimmed
relevance: ""
added: 2026-07-10
last_updated: 2026-07-10
---
<!-- ingested via /paper-search from arXiv (not via Readwise; no rw_id) -->

## TL;DR

CRONOS is an **intervention-based benchmark** that asks whether video generation models predict physical events *consistently* when only nuisance factors of the input change. Built in Unreal Engine with photorealistic rendering (1920×1080, 30 FPS) plus per-object segmentation masks, it holds a physical event fixed (a **fall**, **collision**, or **occlusion**) and systematically varies **four intervention axes** — camera **viewpoint**, **scene**/environment, **object appearance**, and **object category** — then measures how much a model's prediction quality swings. The central construct is **counterfactual physical consistency**: a reliable world model should give the *same* physical rollout under appearance/scene/viewpoint changes (which don't alter dynamics) while adapting coherently when object category changes (which alters mass/friction). Testing four open-source generators (Cosmos2.5 2B/14B, CogVideoX1.5-5B, MAGI-1-4.5B, Wan2.2-14B), the paper finds uniformly low event-success rates (best is Cosmos2.5-2B at **22%**) and, as the headline result, that prediction quality is most fragile to **viewpoint** changes — evidence that current models encode predictions in a "strongly view-dependent way" rather than a stable 3D-aware manner. A surprising secondary finding: scaling Cosmos from 2B to 14B *lowered* success (22% → 14% in V2V).

## Context & motivation

Modern video diffusion models produce visually realistic rollouts and are increasingly pitched as **world models**, but it "remains unclear whether these systems learn underlying causal structure or merely exploit superficial visual correlations for future prediction." Existing physics benchmarks evaluate a model on a *fixed* observation and score whether the generated future looks plausible; they give limited insight into whether the model's *predictive representation* is stable under changes that should be irrelevant to the dynamics (a different camera angle, a recolored object, a different background).

CRONOS reframes the question as one of **robustness to controlled interventions**. The authors argue a reliable predictor should "remain stable under nuisance changes such as viewpoint or appearance variations, while adapting coherently when other aspects of the scene change" (e.g. swapping a light object for a heavy one). By generating counterfactual variants of the *same* underlying event and measuring the spread in prediction quality, the benchmark isolates whether a model has disentangled dynamics from appearance/geometry — a question that fixed-observation benchmarks structurally cannot answer.

## Method

### Problem formulation

Given a conditioning signal (a first frame, or several frames) depicting the start of a physical event, a video model generates a future rollout. CRONOS constructs, for each event, a **family of counterfactual inputs** that differ along exactly one intervention axis while the underlying physical event is held constant. Counterfactual physical consistency is defined as "a model's ability to produce predictions of physical events that remain coherent across counterfactual variants of the visual input." The benchmark operationalizes this as **low variance in per-video quality metrics across the intervention family**.

### Core idea

If the physical event is fixed and only a nuisance factor changes, a model with a genuine dynamics model should produce rollouts of near-identical quality. The *deviation between the best and worst rollout* within an intervention group therefore measures the model's failure to be invariant — the paper's **sensitivity** metric. Low sensitivity = high counterfactual consistency. Crucially, one axis (object category) is *not* a pure nuisance: replacing an object changes both appearance *and* physical parameters (mass, friction), so a good model should *adapt* there, not stay invariant.

### Benchmark construction

- **Engine / rendering**: a controllable Unreal Engine environment with professional 3D assets, indoor and outdoor, rendered at 1920×1080, 30 FPS. The pipeline emits **per-object segmentation masks** alongside RGB, enabling object-centric metrics.
- **Physical events** (3 types):
  - **Fall** — an object rolls across a surface and drops from an edge (contact transition + free-fall).
  - **Collision** — one object impacts another (interaction dynamics, object permanence).
  - **Occlusion** — an object rolls behind a scene element and reappears (temporal coherence / persistence).
- **Four intervention axes**:
  1. **Scene** — background environment and layout change (e.g. the *height* of the surface in a fall), testing adaptation to layout-dependent dynamics.
  2. **Camera viewpoint** — rendering perspective shifts while "keeping scene dynamics intact," probing whether scene geometry is disentangled from observed motion.
  3. **Object appearance** — visual attributes such as color change "without altering physical parameters," isolating appearance-vs-dynamics disentanglement.
  4. **Object category** — the object is "replaced with another compatible object, changing both visual properties and physical parameters" (mass, friction) — the one axis where coherent *adaptation* is expected.
- **Design**: full-factorial combinations of the axes, **except viewpoint is omitted for occlusion events** (to preserve the visibility structure that defines occlusion), yielding **675 videos** across variations.

### Consistency metric

The framework does not define a single closed-form "consistency" scalar; instead it computes **five per-video quality metrics** and then a **sensitivity** statistic over intervention groups.

**Per-video quality metrics** (object-centric, using the propagated masks; masks are initialized from the ground-truth first-frame renders and propagated with SAM3):

- **Appearance / object stability** — DINOv2 CLS-token similarity of each frame to the first frame, background masked out:
  $$\text{ObjectStability} = \text{RobustMin}_{i,t}\big(\langle \text{CLS}(I_i^t)\cdot \text{CLS}(I_i^0)\rangle\big)$$
  where $I_i^t$ is object $i$ at frame $t$, $I_i^0$ its first-frame appearance, $\langle\cdot\rangle$ cosine similarity, and $\text{RobustMin}$ the robust aggregation below.
- **Background stability** — pixel MSE of the shared background region vs. the first frame:
  $$\text{BackgroundStability} = \text{RobustMax}_t\big(\text{MSE}(\text{BG}(I^t), \text{BG}(I^0))\big),\qquad S' = \exp[-50\cdot S]$$
  $\text{BG}(\cdot)$ extracts the background; the exponential rescales the error $S$ into a $(0,1]$ score $S'$.
- **Motion similarity** — cosine similarity of appearance-invariant motion embeddings from **DisMo**, generated video $I$ vs. reference $\tilde I$:
  $$\text{MotionSimilarity} = \text{RobustMin}_t\big(\langle \text{DisMo}(I^t)\cdot \text{DisMo}(\tilde I^t)\rangle\big)$$
- **Shape stability** — Chamfer Distance between the object mesh at frame $t$ and frame $0$, meshes obtained by running **SAM3D** on the generated video:
  $$\text{ShapeStability} = \text{RobustMax}_{i,t}\big(\text{CD}(M_i^t, M_i^0)\big)$$
  with $M_i^t$ the mesh of object $i$ at frame $t$; an exponential rescaling is applied.
- **Physical plausibility** — a VLM judge (**Qwen3-VL-32B**) answers event-specific binary questions $b$; the score penalizes disagreement with the ideal answer:
  $$\text{PhysicalPlausibility} = \Big(1 + \sum_b \mathbb{1}[\,b_{\text{VLM}} \neq b_{\text{Ideal}}\,]\Big)^{-1}$$
  where $\mathbb{1}[\cdot]$ is the indicator, $b_{\text{VLM}}$ the judge's answer and $b_{\text{Ideal}}$ the physically correct answer.

**Robust aggregation** ($\text{RobustMin}/\text{RobustMax}$): rather than a raw min/max over frames (which is noise-sensitive), "compute the average of the worst $k$ frames in the video," with $k \approx 5\%$ of generated frames.

**Sensitivity to interventions** (the counterfactual-consistency measure): for a set of experiments "that differ only along one intervention axis," compute "the deviation between the best and the worst performance" (i.e. best-minus-worst spread of a quality metric across the group), then "average across groups and metrics," reported per intervention type (Scene, Object/category, Appearance, View). **Lower sensitivity is better** — it means "the model's output quality remains stable across controlled interventions," i.e. higher counterfactual consistency.

## Experimental setup

- **Models tested** (open-source only): **Cosmos2.5** (2B and 14B), **CogVideoX1.5-5B**, **MAGI-1-4.5B**, **Wan2.2-14B**.
- **Conditioning settings**: **Image-to-Video (I2V)** — first frame only; and **Video-to-Video (V2V)** — five conditioning frames (V2V run for Cosmos and MAGI-1 only).
- **Metrics**: the five per-video quality metrics above, a physical-event **success rate**, and the per-axis **sensitivity** aggregation.
- **Human validation**: 540 representative videos rated by Prolific annotators on a **1–5 scale** across five dimensions (object appearance, shape, background stability, motion plausibility, event quality); metric thresholds calibrated against human ratings, with alignment confirmed (thresholds set by minimizing the gap between false-positive and false-negative rates / via correlation with human ratings).

## Key results

*Summarized.* Absolute physical-event generation is poor across the board: the **best success rate is 22%** (Cosmos2.5-2B, V2V), with most models in the **1–14%** range. On counterfactual consistency, quality degrades systematically under all four interventions, and the ordering of fragility is the paper's core message:

- **Viewpoint changes produce the largest sensitivity** — the headline finding. Verbatim: "A model with stable 3D-aware predictions should produce similar physical rollouts across viewpoints ... the high observed sensitivity indicates that current models instead encode predictions in a strongly view-dependent way."
- **Appearance** changes cause roughly ~20% quality variation; **scene** changes moderate-to-large; **object-category** changes large.
- Overall conclusion: models "rely on superficial visual correlations rather than producing stable predictions of scene dynamics."

## Ablations

- **V2V vs I2V**: five conditioning frames improve most metrics over first-frame-only conditioning (including background and object stability) — extra frames help build a more robust inference-time representation.
- **Scaling paradox**: scaling Cosmos from **2B → 14B** *decreased* performance, success rate dropping **22% → 14%** in V2V — more parameters did not buy more physical consistency here.

## Limitations

Authors' own: (1) **synthetic-to-real domain gap** — Unreal Engine renders, not real video, limit generalization claims; (2) **single-reference rollouts** — most metrics compare against one reference future despite multiple physically plausible futures being consistent with the conditioning; (3) **limited model scope** — only open-source models; closed systems (Sora, Kling, Veo) not evaluated. [analyst's view] The sensitivity metric depends on the reliability of the auxiliary models it stacks (DINOv2, DisMo, SAM3/SAM3D, Qwen3-VL-32B); errors in mask propagation or mesh extraction propagate into the "physics" score, so some measured sensitivity may reflect perception-model brittleness rather than the generator's dynamics.

## Why it matters [analyst's view]

CRONOS operationalizes a sharp, causal question — *is a video model's physics invariant to the things physics doesn't care about?* — that fixed-observation benchmarks can't touch. The viewpoint result is the interesting one: it suggests today's generators memorize view-conditioned appearance trajectories rather than a 3D dynamics model, which is exactly the failure mode you'd expect if the "world model" framing is aspirational rather than achieved. The counterfactual-invariance framing is a cleaner probe than raw plausibility scoring, and the scaling paradox is a useful caution against assuming physical understanding emerges from scale alone. For the vault this sits squarely alongside the growing cluster of "do video models actually encode physics" papers and gives them a reusable *invariance* axis to compare against.

## Open questions / things to verify

- The exact sensitivity formula (best-minus-worst spread, normalization, weighting across the five metrics) should be confirmed against the paper's equations — the fetched text paraphrases it.
- How much of the measured viewpoint sensitivity is the generator vs. viewpoint-induced error in the auxiliary perception stack (SAM3D meshes, DisMo motion)?
- Would closed models (Sora/Veo/Kling) show the same view-dependence, or is this an open-source artifact?
- The 675-video full-factorial count vs. the 540 human-rated subset — coverage of the human validation across axes.

## Connections

- Contrasts / probes-physics cluster: [[papers/esmati-2026-invisible-hand-physics]] (whether models encode physics vs. surface patterns), [[papers/joseph-2026-physics-video-world-models]], [[papers/yuan-2026-physics-alignment]]
- Related benchmarks: [[papers/lin-2026-phyground]], [[papers/zhao-2026-phyworld]]
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]]

## Selected quotes

> "prediction quality for the same physical event type is affected by appearance, environment, and, particularly by viewpoint changes."

> "A model with stable 3D-aware predictions should produce similar physical rollouts across viewpoints ... the high observed sensitivity indicates that current models instead encode predictions in a strongly view-dependent way."
