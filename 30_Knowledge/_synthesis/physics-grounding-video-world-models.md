---
type: synthesis
title: "Physics Grounding in Video & World Models — vault synthesis"
topic: physics-grounding-video-world-models
papers: [meng-2024-phygenbench, gu-2025-phyworldbench, lin-2026-phyground, begiristain-2026-cronos, zhao-2026-phyworld, yuan-2026-physics-alignment, xiong-2026-physalign, xue-2026-acwm-phys, esmati-2026-invisible-hand-physics, joseph-2026-physics-video-world-models, cao-2026-judgefit, chen-2026-actionable-simulators]
supporting: []
created: 2026-07-22
last_updated: 2026-07-22
---

# Physics Grounding in Video & World Models — vault synthesis

## The picture in brief

Video generators are pitched as world simulators, but every measurement instrument in the vault says the same thing: they render physics-violating scenes while looking photorealistic — [[papers/chen-2026-actionable-simulators]] names this "visual conflation." The cluster splits into three lines: **benchmarks** that quantify the failure (best models: 0.51 PCA, 0.262 joint success, ≤3.3/5, 22% event success), **interventions** that partially fix it (DPO, distillation, inference-time reward — each buying single-digit gains), and **probing** showing the models internally encode physics they fail to render. The frontier question is no longer *whether* models fail physics but *where the knowledge is lost* — and whether the judges we measure with can be trusted at all.

## Main claims by paper

**Benchmarks.** [[papers/meng-2024-phygenbench]] (160 prompts, 27 laws, hierarchical GPT-4o/VQAScore judge PhyGenEval, τ=0.78 vs humans): best model Gen-3 scores only **0.51** on physical commonsense while hitting ~0.80 semantic alignment — physics and semantics are decoupled, and neither scaling nor prompt engineering closes the gap. [[papers/gu-2025-phyworldbench]] (1,050 prompts, 50 phenomena, ICLR'26 oral) adds an **Anti-Physics tier**: models "rationalize" — revert to real physics or freeze the scene rather than follow violating instructions — and best joint success is **0.262** (Pika 2.0); its CAP judge gains +13.5 PC ROC-AUC mostly by telling the MLLM the video is synthetic. [[papers/lin-2026-phyground]] (250 ti2v prompts, 13-law taxonomy, 37.4K human labels) replaces binary scoring with per-law 1–5 Likert: **no model exceeds 3.3/5**, solid-body mechanics is hardest (2.79 vs 3.24 optical), and its open **PhyJudge-9B** cuts judge bias to 3.3% vs Gemini-3.1-Pro's 16.6%. [[papers/begiristain-2026-cronos]] tests *counterfactual consistency* under interventions: prediction quality is most fragile to **viewpoint** changes (models are view-dependent, not 3D-aware), and scaling Cosmos 2B→14B *dropped* success 22%→14%. [[papers/xue-2026-acwm-phys]] adds the action-conditioned axis (8 sim environments, 4 physics categories): OoD degradation tracks **task complexity/DoF, not physics category** (Robot Arm ΔM-MSE +40.35 vs stable Push Cube) — evidence for appearance statistics over internalized physics.

**Interventions.** [[papers/zhao-2026-phyworld]]: physics violations as a *preference-learning* problem — rank-16 LoRA DPO (β=100, high-noise window t∈[901,999]) over 1,000 human physics-preference pairs on Wan2.2-I2V lifts its physics benchmark 2.99→**3.09** and VBench to 0.769. [[papers/xiong-2026-physalign]]: train-time distillation — Gram-matrix relational alignment to a frozen V-JEPA2 teacher plus a 4-term depth loss, trained on 3K Blender rigid-body clips; PhysicsIQ 29.6→**38.1%** with *improved* VBench and zero inference cost. [[papers/yuan-2026-physics-alignment]]: no training at all — V-JEPA2 predictive "surprise" (cosine mismatch of predicted vs actual future latents) as a differentiable reward for guidance+Best-of-N; PhysicsIQ V2V 55.2→**62.0%** (+6.78 over SOTA, ICCV'25 challenge winner). Its reward comparison is damning for VLM judges: Qwen-VL rewards score near chance.

**Probing.** [[papers/esmati-2026-invisible-hand-physics]]: inverting the flow-matching sampler recovers latent trajectories for *real* videos; linear probes on DiT states decode plausibility at **81.27%** (WAN-1.3B) — beating V-JEPA2 (71.36%) — while VAE latents sit at chance. The signal lives in the *continuous ODE trajectory* (20-step inversion still reconstructs the video but collapses probe accuracy 0.82→0.57). [[papers/joseph-2026-physics-video-world-models]]: video encoders show a sharp mid-network "Physics Emergence Zone"; direction is a circular population code, not a factorized variable — no classical physics engine inside. _(vault note is abstract-only — thin.)_ [[papers/cao-2026-judgefit]]: VLM physics-judges disagree with humans *and each other*; fitting a per-VLM taxonomy lifts mean Spearman ρ 0.239→0.315 (+32%) across 16 VLMs — but the best judge still only reaches ρ=0.484.

## Comparison

| | Object measured | Scoring | Judge | Headline |
|---|---|---|---|---|
| PhyGenBench | T2V, 27 laws | 0–3 discretized | GPT-4o pipeline | best 0.51 |
| PhyWorldBench | T2V + anti-physics | binary SA/PC | closed MLLM + CAP | best 0.262 |
| PhyGround | ti2v, per-law | 1–5 Likert | humans + open 9B | best 3.28/5 |
| CRONOS | invariance under interventions | sensitivity spread | metric stack + VLM | best 22% |
| ACWM-Phys | action-conditioned InD/OoD | M-MSE/SSIM | pixel metrics | OoD tracks DoF |

**Formulation.** The benchmarks disagree on what "knowing physics" means: outcome adherence (PhyGenBench/PhyGround), instruction-following under counterfactual laws (PhyWorldBench's anti-physics), invariance to nuisance factors (CRONOS), or action-conditioned rollout fidelity (ACWM-Phys). These are progressively stricter: a model could pass outcome tests by memorizing typical trajectories yet fail CRONOS's viewpoint intervention — which is exactly what happens.

**Judging is the contested ground.** PhyWorldBench accepts closed MLLM judges at 75.1 PC ROC-AUC; PhyGround explicitly rejects them (drift, bias) and fine-tunes an open judge; JudgeFit shows even fitted VLM judges cap at ρ≈0.48; WMReward finds VLM rewards near chance while latent-WM surprise works. Convergent conclusion from three independent papers: **VLMs are weak physics perceivers; predictive latent models are better probes.** But Esmati complicates this: the generator's *own* DiT states out-probe V-JEPA2 — the best physics representation may already be inside the model being judged.

**Where the failure lives.** PhyGenBench blames training data (calls for synthetic data); PhysAlign agrees and manufactures it; PhyWorld treats it as a preference/output-distribution problem; Yuan et al. argue it's substantially a *sampling* problem; Esmati provides the mechanism for both post-hoc views: physics is encoded internally but not expressed in outputs. The interventions' magnitudes are consistent with this — all three (DPO +0.10 Likert, distillation +8.5 PhysicsIQ, inference-time +6.8 PhysicsIQ) recover real but bounded gains, as expected if they surface latent knowledge rather than add it.

**A genuine disagreement on scale.** CRONOS: Cosmos 2B→14B *hurts* physical success; ACWM-Phys: DiT-S→DiT-M improves both InD and OoD; PhyGenBench: scaling helps statics, fails dynamics. Not directly comparable (pretrained generalist vs task-trained predictor), but the vault currently holds contradictory evidence that scale buys physics.

## Synthesis [analyst's view]

The three lines triangulate one thesis: **the representation is ahead of the generation.** Diffusion transformers demonstrably build linearly decodable physical structure (Esmati; Joseph's emergence zone), yet outputs fail benchmarks because nothing in the training objective or sampler makes that structure load-bearing. All three intervention papers are, mechanically, ways of coupling a physics-sensitive signal back into generation — DPO via human preferences, PhysAlign via a V-JEPA2 teacher, WMReward via V-JEPA2 surprise. Notably, two of three route through V-JEPA2 even though Esmati shows the generator's own internals carry a *stronger* signal — self-guidance from DiT internal states is the obvious unexplored combination, and Esmati gestures at exactly that. Meanwhile the measurement layer is shakier than the leaderboards suggest: JudgeFit's ρ≤0.48 ceiling and PhyGround's 16.6% Gemini bias imply that cross-benchmark comparisons (0.262 vs 0.51 vs 3.3/5) reflect judge choice as much as model capability. PhyGround's human-anchored per-law protocol plus PhysicsIQ (the one metric shared by three method papers) are currently the most trustworthy yardsticks. What would settle the field's central question is a CRONOS-style intervention test applied to an aligned model: if DPO/WMReward gains survive viewpoint changes, physics was genuinely surfaced; if not, the alignment is another appearance shortcut.

## Gaps

- [[papers/joseph-2026-physics-video-world-models]] — _vault note too thin — deep-read needed_ (abstract-only; encoder list, probe details, benchmarks all missing).
- Recurring citations with no vault note: **Garrido et al. 2025** (intuitive physics from V-JEPA — the empirical basis for WMReward and PhysAlign's teacher choice) _needs note_; **PhysicsIQ** benchmark paper (the shared metric of three method papers) _needs note_; **VideoPhy / VideoPhy-2** (JudgeFit's calibration data) _needs note_; **VideoJAM / NewtonGen / WISA** (train-time physics injection baselines) _needs note_; **PIN-WM** (explicit differentiable-physics anchoring, the far end of Chen & Zhu's spectrum) _needs note_.
- No paper in the vault yet tests whether alignment gains transfer to *downstream control* — the closed-loop evaluation [[papers/chen-2026-actionable-simulators]] demands.

## Sources

- [[papers/meng-2024-phygenbench]] — core (seminal benchmark)
- [[papers/gu-2025-phyworldbench]] — core (anti-physics tier)
- [[papers/lin-2026-phyground]] — core (per-law human study, open judge)
- [[papers/begiristain-2026-cronos]] — core (counterfactual invariance)
- [[papers/xue-2026-acwm-phys]] — core (action-conditioned axis)
- [[papers/zhao-2026-phyworld]] — core (train-time DPO)
- [[papers/xiong-2026-physalign]] — core (train-time distillation)
- [[papers/yuan-2026-physics-alignment]] — core (inference-time reward)
- [[papers/esmati-2026-invisible-hand-physics]] — core (internal probing)
- [[papers/joseph-2026-physics-video-world-models]] — supporting (thin note)
- [[papers/cao-2026-judgefit]] — contrast (judge reliability)
- [[papers/chen-2026-actionable-simulators]] — framing (position survey)
- [[topics/world-models]], [[topics/video-generation]]
