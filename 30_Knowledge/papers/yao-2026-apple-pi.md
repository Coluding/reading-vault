---
type: paper
title: "Apple-π: Benchmarking Thinking with Video Towards Law-Grounded Physical Intelligence"
authors: ["Runmao Yao", "Kairui Hu", "Yukang Cao", "Ruisi Wang", "Shulin Tian", "Ziang Cao", "Weichen Fan", "Ziqi Huang", "Yuhao Dong", "Hao Li", "Zhaoxi Chen", "Zhongang Cai", "Lei Yang", "Ziwei Liu"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2607.16401
rw_id: 01ky7gc3m6ff0jn3shssysrkyc
topics: [world-models, video-generation, reasoning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# Apple-π: Benchmarking Thinking with Video Towards Law-Grounded Physical Intelligence

## TL;DR

Apple-π (S-Lab NTU + CUHK) is a benchmark claiming to be the first that anchors video-model evaluation explicitly in physical laws rather than in output-level plausibility. It decomposes physical reasoning into three auditable stages — **Perception** (read the physical quantities and objects), **Formulation** (identify the governing law), **Deduction** (roll the law forward into dynamics) — and forces every answer to be delivered *as a video*, treating the generated clip as a visible "chain-of-frames" reasoning trace. The substrate is **Orchard**, 400 classical-mechanics videos (243 Isaac Sim, 121 self-recorded lab, 36 internet physics-education clips) spanning ten canonical tasks in three single-law pillars (gravitation, momentum conservation, Newton's first law) plus multi-law compositions; every case carries exact physical parameters overlaid as infographic annotations on the first frame. Scoring is a hybrid of MLLM-judge rubrics (Gemini 3 Flash, temp 0) and physics-derived objective metrics (mask IoU, three Physics-IQ-style motion-mask IoUs, 3D velocity error), with the Deduction score weighting physics at 0.60. Across 11 models, the best pure video model (Seedance 2.0) scores only **0.473**, while unified understanding-generation models GPT Image 2 (**0.704**) and Nano Banana 2 (**0.699**) lead overall — yet even they hit only ~0.40 on Deduction. The headline diagnostic findings: a Perception → Formulation → Deduction "reasoning funnel", weak state transfer in multi-law compositions, and a persistent Sim-to-Real gap.

## Context & motivation

Video generation models are increasingly framed as emerging world models — Sora's "video generation as world simulation" framing, plus recent evidence that video models can act as zero-shot reasoners via **chain-of-frames** generation (Wiedemer et al.'s explicit analogy between frame-by-frame video generation and step-by-step textual reasoning) and "thinking with video". The paper's framing device: current models resemble *Aristotle* (intuitive descriptions absorbed from data), and the open question is whether they can reason like *Newton* (perceive quantities → abstract a law → deduce consequences).

The gap the authors identify in prior benchmarks: existing physical-intelligence benchmarks — VideoPhy-2, PhyWorldBench, PhyGenBench, Physics-IQ, WorldModelBench, VideoScience-Bench, PhysicsMind, plus understanding-side suites like PhysBench, QuantiPhy, PhyX — evaluate only **what a model outputs, never how it got there**. When a model succeeds, one cannot tell whether it invoked a law or merely produced something that looks right; when it fails, one cannot tell whether it misread the scene, misidentified the law, or stumbled in the deduction. Per the paper's Table 1, Apple-π is the only entry with full support for all four of: law grounding, multi-stage evaluation, chain-of-frames protocol, and objective physics metrics (PhyWorldBench and PhysicsMind get only partial law-grounding; Physics-IQ has objective metrics but no stage decomposition).

## Method

### Problem formulation

Input per case: an **infographic-annotated first frame** (physical parameters overlaid next to their visual referents — environmental constants like $g$ as global labels, object properties like mass adjacent to objects) plus a subtrack-specific chain-of-frames text prompt. Output: always a generated video (or, for unified models, still images / timestamped keyframes). The object being measured is not video quality per se but whether the model's visible reasoning trace is consistent with the governing physical law. Each subtrack score lives on $[0,1]$.

### Core idea

Turn Newton-style scientific reasoning into an auditable video-generation protocol: split reasoning into Perception / Formulation / Deduction stages, elicit each as a video whose final frame (or full trajectory) *is* the answer artifact, and score against law-predicted ground truth — so failures can be localized to a stage instead of collapsed into one plausibility number.

### Orchard dataset

400 cases from three complementary sources:

- **Simulated (243)**: NVIDIA Isaac Sim, deterministic configurations, programmatic ground truth (RGB, masks, depth, poses, velocities, contact events read from engine state). Two authoring consistency rules: mass derived from material density and analytic primitive volume ($m = \rho V$), and a single material key shared between visual and physics material so appearance and dynamics can't diverge. Closed-form analytic solutions (e.g. ideal collision velocities, projectile states) serve as the canonical answer key when available.
- **Self-recorded real-world (121)**: controlled lab conditions at 30 fps; parameters instrumented (mass ±0.01 g on a digital balance, size ±0.05 mm caliper, drop height ±1 mm, initial velocity ±0.05 m/s via frame-by-frame velocimetry, restitution ±0.02 via two-drop ratio test, friction ±0.05 via tilt-threshold test).
- **Internet-sourced (36)**: physics-education YouTube channels, native fps kept in metadata.

Object identity is controlled as a nuisance variable: the object vocabulary is standardized to four primitive solids (sphere, cube, cylinder, cone), whose simple geometry gives well-defined centers, masks, and contact surfaces, and reduces the chance that performance is driven by object-specific semantic priors.

**Task taxonomy** (two levels). Top level splits **single-law** (confounder-free diagnosis) vs **multi-law** (compositional generalization). Single-law has three pillars: (1) *universal gravitation* — free fall ($h = \tfrac{1}{2}gt^2$), projectile motion ($\vec r(t) = \vec r_0 + \vec v_0 t + \tfrac{1}{2}\vec g t^2$), inclined plane ($a = g\sin\theta$), gravity-driven circular motion ($mg = mv^2/r$); (2) *conservation of momentum* — perfectly elastic, perfectly inelastic, and inelastic collisions, parameterized by coefficient of restitution $e = 1$, $e = 0$, $0 < e < 1$; (3) *Newton's first law* — objects at rest and uniform linear motion under $\sum \vec F = 0$. The multi-law branch chains these (e.g. inclined plane feeding into projectile motion; circular motion followed by a collision), specifically probing whether the state output of one law becomes the correct initial condition of the next.

Quality assurance: a 1+2 three-pass review protocol (one annotation pass, two independent reviews, disagreements resolved by analytic backstop → senior arbitration → domain-expert escalation), semi-automatic mask annotation with human correction and re-propagation, and a retroactive inter-annotator agreement study (Cohen's κ 0.89–1.00 on categorical fields; ICC(2,1) 0.88–0.99 on continuous fields).

### Benchmark protocol: five subtracks

Perception and Formulation each split into *text* and *graphic* subtracks; Deduction is a single track:

1. **Perception-Text** (OCR analogue): reproduce the numeric annotations from the input frame as a *final-frame artifact with fade-to-white* — the video fades everything except the annotations to a white background, preserving each annotation at its original spatial position.
2. **Perception-Graphic** (instance-segmentation analogue): localize the experiment-relevant objects; environment fades to white, target objects preserved unchanged.
3. **Formulation-Text** (symbolic law selection): four-option multiple choice with distractors engineered to expose specific failure modes — a *confusing* real law sharing symbols with the annotations, an *unrelated* real law, and a *fabricated* formula that doesn't exist in physics. (A model that picks by superficial symbol matching should fail on the confusing distractor.) Answer: three lines on white — option letter, symbolic formula, and the formula with annotated symbols numerically substituted.
4. **Formulation-Graphic** (grounded state prediction): given a target instant $t^\star$ in the prompt, render the scene at $t^\star$ *without* fade-to-white — each object at its predicted position, overlaid with a borderless orange velocity arrow from its center and a "v = X.XX" speed label.
5. **Deduction**: generate the full video of the scene evolving under the governing law for the requested physical duration $T_c$; evaluated frame-by-frame against the law-predicted trajectory.

Unified understanding-generation models emit PNGs directly for tracks 1–4 and timestamped keyframes at the evaluation timestamps for Deduction; video models emit MP4s for everything, with the last frame taken as the answer artifact on tracks 1–4. Prompts are not tuned per model.

### Evaluation suite (scoring definitions)

**Time alignment.** Physical time is defined by the prompt, not the provider's container duration. With $N^{\text{gen}}$ decoded frames and requested duration $T_c$, the effective fps is $\hat f^{\text{gen}} = N^{\text{gen}}/T_c$; the generated frame nearest physical time $t$ is $j_c(t) = \mathrm{clip}(\mathrm{round}(t \hat f^{\text{gen}}), 0, N^{\text{gen}}{-}1)$, matched to the GT frame $i_c(t) = \mathrm{clip}(\mathrm{round}(t f^{\text{GT}}_c), 0, N^{\text{GT}}_c{-}1)$. All pixel/mask metrics computed at GT resolution (bilinear resize for RGB, nearest-neighbor for masks). Each case has a canonical timestamp set $\mathcal T_c$ that includes event-critical moments (collision contact, ramp exit) beyond the uniform grid.

**MLLM-judge rubrics.** Gemini 3 Flash at temperature 0, subtrack-specific rubrics with criteria in $[0,1]$ grouped into semantically coherent groups, weighted-averaged per subtrack (e.g. Perception-Text: content 0.50 / layout 0.30 / style 0.20 across 18 criteria; P-G and F-G use group weights 0.10/0.30/0.30/0.30). Inapplicable criteria score −1 and are excluded with weight renormalization; undecodable rollouts score 0.

**Objective physics metrics.** For graphic subtracks, per-object mask IoU $\mathrm{IoU}_o = |M_o^{\text{gen}} \cap M_o^{\text{GT}}| / (|M_o^{\text{gen}} \cup M_o^{\text{GT}}| + \epsilon)$, with Hungarian matching over center distance and bbox overlap when identity is unavailable and missing objects scored zero. For Deduction, following Physics-IQ, SAM3 extracts binary motion masks $A_t^x(p) \in \{0,1\}$ (pixel $p$ belongs to a law-relevant moving object at time $t$) for both generated and GT clips, and three IoUs are computed:

- **Spatial IoU** ("where does action happen?"): collapse time via $S^x(p) = \max_{t \in \mathcal T_c} A^x_t(p)$, then $S_{\text{spatialIoU}} = \frac{\sum_p \mathbf 1[S^{\text{gen}}(p) \wedge S^{\text{GT}}(p)]}{\sum_p \mathbf 1[S^{\text{gen}}(p) \vee S^{\text{GT}}(p)] + \epsilon}$ — timing-agnostic spatial support of motion.
- **Spatiotemporal IoU** ("where *and when*?"): frame-wise IoU at each aligned timestamp, averaged over $\mathcal T_c$ — high Spatial but low Spatiotemporal IoU means right places, wrong times.
- **Weighted Spatial IoU** ("where and *how much*?"): temporal-average occupancy $W^x(p) = \frac{1}{|\mathcal T_c|}\sum_t A^x_t(p)$, compared by the generalized ratio $S_{\text{wIoU}} = \sum_p \min(W^{\text{gen}}, W^{\text{GT}}) / (\sum_p \max(W^{\text{gen}}, W^{\text{GT}}) + \epsilon)$ — distinguishes transient from repeated/prolonged motion at the same location.

Plus **velocity accuracy**: with mean Euclidean 3D velocity error $e_v$ against the law-predicted velocity, $S_{\text{vel}} = 1/(1 + e_v)$, and PSNR-based fidelity terms $S_{\text{PSNR}} = \min(\text{PSNR}/40, 1)$ (also a masked variant on the union of GT foreground masks).

**Deduction fusion.** Three groups — integrity $= \tfrac12(S_{\text{ann\_removed}} + S_{\text{obj\_consistency}})$; fidelity $= \tfrac14(S_{\text{visual}} + S_{\text{smoothness}} + S_{\text{PSNR}} + S_{\text{maskedPSNR}})$; physics $= \tfrac15(S_{\text{phys\_acc}} + S_{\text{spatialIoU}} + S_{\text{stIoU}} + S_{\text{wIoU}} + S_{\text{vel}})$ — combined as

$$S_{\text{Ded}} = 0.20\, S_{\text{integrity}} + 0.20\, S_{\text{fidelity}} + 0.60\, S_{\text{physics}},$$

the 0.60 physics weight expressing that visual plausibility must not compensate for physically incorrect dynamics. Scores aggregate rollout (mean of 3) → case → slice → model, with all reported cells case-level means over the relevant subset.

_No derivation beyond the metric definitions; benchmark paper._

### Judge reliability

An open-weights cross-check re-evaluates a balanced subset with Qwen3-VL-30B-A3B-Instruct-FP8: Pearson r = 0.95 / Spearman ρ = 0.93 on overall model averages, with per-track agreement 0.71–1.00 (weakest on Perception-Graphic). Residual disagreement: the open judge is more lenient toward visually plausible law-violating motion and scores rubric items more independently. Tier-level rankings are preserved, so Gemini 3 Flash stays canonical.

## Experimental setup

- **Models (11)**: video generation — Wan2.2, HunyuanVideo-1.5, VBVR-Wan2.2 (Wan2.2 fine-tuned on the video-reasoning VBVR-Dataset), Seedance 2.0, Veo 3.1; unified understanding-generation — BAGEL, OmniGen2, SenseNova-U1-8B-MoT (± Think), GPT Image 2, Nano Banana 2.
- **Scale**: 400 cases × 5 subtracks × 3 rollouts = 6000 evaluated responses per model.
- **Metrics**: as above; results sliced by track (P-T, P-G, F-T, F-G, Ded.), pillar (Grav., Mom., N1, Multi), and source (Sim., Real).

## Key results

- **Overall**: GPT Image 2 **0.704**, Nano Banana 2 **0.699** lead; best video model Seedance 2.0 **0.473**; VBVR-Wan2.2 0.373, SenseNova-U1 ~0.36, Veo 3.1 0.313, Wan2.2 0.267, BAGEL 0.218, OmniGen2 0.214, HunyuanVideo-1.5 0.177.
- **Deduction is the universal bottleneck**: even the strongest unified models score only ~0.40–0.41 on Deduction (GPT Image 2 0.406, Nano Banana 2 0.405); video models range 0.149–0.315.
- **Reasoning funnel**: for video models, scores decline Perception → Formulation → Deduction; earlier-stage success is necessary but not sufficient (copying annotations ≠ identifying the law; selecting a law ≠ executing it over time).
- **Subtrack failure structure**: Perception-Text easier than Perception-Graphic (reading quantities beats grounding them to regions); Formulation-Text vs Formulation-Graphic exposes the gap between symbolic law selection and grounded state prediction. Striking case: Wan2.2 and VBVR-Wan2.2 score 0.009 / 0.001 on Formulation-Text — near-total failure at formula selection — while Seedance 2.0 gets 0.478 and GPT Image 2 0.824.
- **Sources of capability**: proprietary video models beat open-source base generators (large-scale video training distills physical priors); VBVR-Wan2.2's reasoning fine-tuning lifts Perception (P-T 0.923) but doesn't transfer to Formulation or Deduction. Strong closed unified models beat all video models especially on Perception/Formulation, but open unified models are much weaker — so the advantage isn't the unified interface alone; the paper attributes it to stronger multimodal foundations plus post-training for instruction following, text rendering, and layout control.
- **Multi-law < single-law** across models (e.g. Seedance 2.0: 0.389 Multi vs 0.450–0.510 single pillars) — weak carrying of physical state across law transitions.
- **Sim-to-Real gap**: all models score lower on real than simulated cases (e.g. Seedance 0.487 → 0.459; Wan2.2 0.310 → 0.224); since laws are unchanged, the drop reflects grounding/tracking failure under realistic visual conditions.
- **Qualitative failure modes**: annotation-semantics failures (model preserves text and tracks the object but misbinds e.g. an initial-velocity arrow's direction as a physical initial condition, corrupting the target state and trajectory), and pre-physics failures (blurred annotation text, failure to isolate objects, unreadable formulas, missing velocity arrows — OCR/segmentation/rendering/instruction-following limits).

## Ablations / validation studies

Two protocol ablations (paired deltas on Veo 3.1 and Nano Banana 2) verify the protocol doesn't inject solutions:

- **Text-Parameter** (quantities as structured text instead of infographic overlays): near-neutral, avg Δ +0.008 / +0.020, all track deltas within ±0.05 — the benchmark is *not* bottlenecked by reading infographics; annotations are kept as a spatially grounded, auditable binding of quantities to referents, not a shortcut.
- **Concise-Prompt** (drop detailed output-format instructions): slight average drop (−0.018 / −0.037), concentrated on graphic tracks (P-G up to −0.085, F-G up to −0.096) while Formulation-Text *improves* +0.051 for both — the full prompt standardizes visual answer format rather than supplying physics.

Also validation-flavored: the Qwen3-VL judge cross-check (above) and the simulator QA pass that caught 63 issues (23 physics-duration adjustments, 10 velocity-label overrides, 30 re-recordings).

## Limitations

Paper's own (Appendix H): scope restricted to classical rigid-body mechanics (no fluids, thermodynamics, EM, deformables, fracture, granular media); objects restricted to four primitives with a single fixed camera (no articulated/deformable objects, no multi-view); the annotated first frame means it does not test constructing a scene from text alone; MLLM judges are not physical oracles and objective metrics inherit segmentation/pixel/velocity-estimation noise; real-world ground truth is measurement-based and less exact than simulator state; and the temporal-normalization rule *assumes* the generated clip represents the requested physical duration rather than the provider's container length.

[analyst's view] Additional flags: (1) the Deduction comparison between families isn't fully apples-to-apples — unified models are scored on sparse keyframes with object-position-match proxying motion smoothness, which plausibly flatters them on the hardest track; (2) heavy reliance on format compliance (fade-to-white, orange arrows, three-line answers) means part of every score measures controllable rendering rather than physics, which the concise-prompt ablation partially but not fully disentangles; (3) 36 internet cases is a thin slice for the Sim-to-Real claim.

## Why it matters [analyst's view]

This is the most process-oriented entry in the vault's physics-benchmark cluster: where [[papers/meng-2024-phygenbench]], [[papers/gu-2025-phyworldbench]] and Physics-IQ-style suites score whether the *output* looks physically right, Apple-π asks the model to externalize each reasoning stage as a video artifact and localizes failure to Perception vs Formulation vs Deduction. That stage-resolution is genuinely new evaluative information — the finding that VBVR-style reasoning fine-tuning fixes Perception but not Formulation/Deduction, and that even frontier unified models cap at ~0.40 on Deduction, is a sharper diagnosis than "model X is 20 points more physical than model Y". The strong showing of unified understanding-generation models supports a thesis running through this cluster: physical correctness may need explicit understanding machinery, not just bigger video pretraining. The "chain-of-frames as auditable reasoning trace" framing also connects the physics-benchmark line to the thinking-with-video literature — the benchmark effectively operationalizes Wiedemer et al.'s analogy as an evaluation protocol. Worth watching whether the fade-to-white answer-artifact convention gets adopted; it is a clever way to make a generative model emit discrete answers, but it also entangles physics scores with rendering control.

## Open questions / things to verify

- How much of the video-model Formulation-Text collapse (Wan2.2 at 0.009) is inability to *render text* vs inability to *select the law*? The concise-prompt ablation touches this but a text-output control would settle it.
- Does the keyframe protocol for unified models on Deduction systematically inflate their physics-group scores relative to dense-video scoring?
- Per-distractor breakdown on Formulation-Text (confusing vs unrelated vs fabricated) is not reported in the main text — that split would directly test the symbol-matching hypothesis.
- How sensitive are the SAM3 motion masks (and hence 3 of 5 physics-group terms) to segmentation drift on generated videos with degraded object appearance?
- VBVR-Wan2.2's dataset ("A Very Big Video Reasoning Suite", Wang et al. 2026) as a candidate read for what reasoning supervision does and doesn't buy.

## Connections

- Contrasts with: [[papers/meng-2024-phygenbench]] — cited by the paper (Table 1: T2V, 160 cases, physical commonsense, no law grounding / stages / objective physics metrics); Apple-π positions itself as adding exactly those.
- Contrasts with: [[papers/gu-2025-phyworldbench]] — cited (Table 1: T2V, 700 cases, mechanics, partial law grounding, no stage decomposition or chain-of-frames).
- Related (vault-adjacent, not cited by the paper): [[papers/lin-2026-phyground]] — physical reasoning in generative world models; [[papers/begiristain-2026-cronos]] — counterfactual physical consistency in video models; [[papers/xue-2026-acwm-phys]] — physical interaction in action-conditioned video world models.
- Related (vault-adjacent, not cited): [[papers/cao-2026-judgefit]] — per-VLM judge taxonomies for physical video evaluation; directly relevant to Apple-π's MLLM-judge reliability question and its Gemini-vs-Qwen3-VL cross-check.
- Synthesis: [[_synthesis/physics-grounding-video-world-models]]
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/reasoning]]

## Selected quotes

> "Yet existing benchmarks largely evaluate physical plausibility only at the output level, without verifying whether the model arrives there through a faithful, law-grounded reasoning process." — Abstract

> "can video models reason about the physical world in a law-grounded manner, like Newton, rather than relying on intuition, like Aristotle?" — §1

> "The larger physics weight reflects the benchmark objective: visual plausibility should not compensate for physically incorrect dynamics." — §E.3

> "This shows that copying annotations does not imply understanding the underlying physical variables." — §4.5

> "Moreover, even the strongest unified models score only around 0.40 on Deduction, indicating that law-consistent temporal dynamics remains the central bottleneck." — §4.2
