---
type: synthesis
title: "Multimodal & Sensory World Models — vault synthesis"
topic: multimodal-sensory-world-models
papers: [higuera-2026-visuo-tactile-world-models, ivashkov-2026-sensorimotor-world-models, shang-2025-roboscape, ye-2026-world-action-models, tong-2026-beyond-language-modeling, mazzaglia-2024-genrl, zuo-2026-qwen-agentworld, zhou-2024-robodreamer, karcini-2026-robots-beyond-vla, ding-2024-world-models-survey, yao-2026-apple-pi]
supporting: []
created: 2026-07-23
last_updated: 2026-07-23
---

# Multimodal & Sensory World Models — vault synthesis

## The picture in brief

Every paper here adds a channel beyond RGB to a world model, but they disagree on what the channel *is* and where it lives: **sensory grounding** (touch, depth, keypoints), **action as a modality** (inverse dynamics, joint video–action denoising), or **semantic coupling** (frozen VLM embeddings, parsed language, pure text). The strongest shared finding is ablation-shaped — remove the extra channel and performance collapses far beyond what "auxiliary loss" suggests ([[papers/mazzaglia-2024-genrl]]'s aligner: 0.76 → 0.17). The frontier question is whether the second modality must be *present at inference* or need only *shape the representation during training*; most of the vault says the latter.

## Main claims by paper

**Sensory channels.** [[papers/higuera-2026-visuo-tactile-world-models]] (VT-WM) makes tactile load-bearing for contact physics: 33% better object permanence, 29% better motion-law compliance, up to 35% higher zero-shot real-robot success — but the note is abstract-only, so fusion mechanism and baseline are unknown. [[papers/shang-2025-roboscape]] *derives* its extra channels from RGB instead of sensing them: a depth branch fused per-block into the RGB branch ($\mathbf{h}^{l}_{\text{RGB}} \leftarrow \mathbf{h}^{l}_{\text{RGB}} + \mathcal{W}^{l}(\mathbf{h}^{l}_{\text{depth}})$) plus keypoint token-consistency along tracked trajectories; ablating both drops action controllability ΔPSNR 3.34 → 1.99, and it hits r = 0.953 as a policy evaluator.

**Action as the modality.** [[papers/ivashkov-2026-sensorimotor-world-models]] (SMWM) makes a single inverse-dynamics head the *sole* anti-collapse mechanism, $\mathcal{L} = \mathcal{L}_{\text{fwd}} + \lambda\mathcal{L}_{\text{inv}}$: a collapsed encoder can do no better than predicting $\mathbb{E}[a_t]$, so nontrivial inverse prediction rules collapse out — and it concentrates latents on controllable DoF (PCA rank matches action dimension exactly; 84% vs 59% for SIGReg on OGBench-Cube). [[papers/ye-2026-world-action-models]] (DreamZero, 14B) denoises a joint $[\mathbf{z}, \mathbf{a}]$ velocity field under the factorization $\pi(\mathbf{o},\mathbf{a}) = \pi(\mathbf{o})\cdot\pi(\mathbf{a}\mid\mathbf{o})$ — video prediction × IDM — reaching 62.2% vs 27.4% task progress, with cross-embodiment transfer from *action-free* video (38.3% → 55.4% robot, → 54.3% human egocentric).

**Semantic coupling.** [[papers/mazzaglia-2024-genrl]] bolts a frozen InternVideo2 onto a vision-only world model via a connector (VLM embedding → latent trajectory) plus an aligner closing the vision–language gap; 0.80 vs 0.70 best baseline across 35 reward-free tasks. [[papers/zhou-2024-robodreamer]] factorizes on the *condition* side, $p_\theta(\tau\mid L) \propto \prod_i p_\theta(\tau\mid l_i)^{1/N}$, making novel instructions recombinations of in-distribution parts (81.3% vs 68.8% unparsed on unseen tasks; goal image → 95.8%). [[papers/tong-2026-beyond-language-modeling]] argues no such machinery is needed — world modeling *emerges* under joint pretraining with actions as plain text tokens, saturating at ~1% in-domain data — while measuring a scaling asymmetry ($b \approx 0.63$ vision vs $0.53$ language). [[papers/zuo-2026-qwen-agentworld]] is the limit case of text as the *only* modality; controllable Sim RL inside it beats a live search engine (50.3% vs 45.6% F1-item).

**Framing.** [[papers/karcini-2026-robots-beyond-vla]]: the bottleneck is neither policy nor modality but the missing machinery turning unstructured multimodal experience into grounded supervision — latent transition codes $z_t \sim q(\cdot\mid o_t, o_{t+1}, L_t, L_{t+1})$ that become actions only under an embodiment-conditioned decoder. [[papers/ding-2024-world-models-survey]] names multimodal integration one of four capabilities separating video world models from video generators. [[papers/yao-2026-apple-pi]] measures the payoff: unified understanding-generation models score 0.704/0.699 vs 0.473 for the best pure video model, yet all cap near 0.40 on Deduction.

## Comparison

| Paper | Extra channel | Fusion mechanism | Needed at inference? |
|---|---|---|---|
| VT-WM | tactile (sensed) | _unknown — note too thin_ | yes |
| RoboScape | depth + keypoints (derived) | per-block additive feature injection | no (annotators dropped) |
| SMWM | action | auxiliary inverse head, $\lambda\mathcal{L}_{\text{inv}}$ | no (head discarded when planning) |
| DreamZero | action + language + proprio | joint $[\mathbf{z},\mathbf{a}]$ velocity field | yes |
| GenRL | video–language (frozen VLM) | connector + aligner into latent space | yes (prompt only) |
| RoboDreamer | parsed language, goal image/sketch | product-of-experts score averaging | yes (as conditions) |
| Tong et al. | text + image + video + action | modality-specific FFNs → MoE routing | yes |
| Qwen-AgentWorld | text only | single trajectory schema | n/a |

**Formulation.** The deepest split is *what the second modality is for*. VT-WM and DreamZero treat it as **information** the model otherwise lacks (contact state; the action to execute). RoboScape and SMWM treat it as a **regularizer** — a training-time constraint shaping the representation, then discarded. These aren't competing answers to one question: derived channels only re-express what pixels already contain, whereas touch is genuinely new information under occlusion. That distinction is exactly what the thin VT-WM note prevents the vault from adjudicating.

**Where alignment bites.** GenRL's aligner ablation is the sharpest cross-cutting result: removing it collapses GenRL 0.76 → 0.17, but adding it to baselines using VLM embeddings for *similarity scoring* changes nothing (0.40 → 0.42). Modality-gap alignment matters enormously when an embedding is an **input to another network**, barely at all when it is a scalar reward — a rule transferable to any bolt-on-a-foundation-model design.

**Disagreements.** (1) *Emergence vs structure*: Tong claims actions-as-text plus generic pretraining suffices; RoboDreamer, GenRL and SMWM each build explicit machinery and each show ablating it is catastrophic. Not directly comparable — Tong's world-modeling is navigation (its own note flags this as narrow) while the structural papers test contact-rich control — so the vault holds both "scale dissolves the problem" and "the mechanism is load-bearing" with no common benchmark; Karcini rejects both as insufficient. (2) *Is text enough?* Qwen-AgentWorld's text-only model beats every frontier model overall (58.71) yet ranks 5th on GUI, which its note attributes to missing multimodal pretraining; Apple-π independently finds understanding-heavy unified models beat pure video generators. Both say semantic machinery helps — the GUI gap marks where text-only representation goes lossy.

## Synthesis [analyst's view]

The combined evidence supports a claim none of these papers makes alone: **the second modality is usually doing representational work, not informational work.** RoboScape's depth is computed from its own RGB; SMWM's actions are already in the dataset; RoboDreamer's parse adds no content to the instruction. In each case the gain comes from *constraining what the representation must preserve* — which is why the channel can be dropped at inference and the gains persist. That reframes multimodality here as inductive bias, and predicts VT-WM should be the outlier, since occluded contact state is information no camera provides — but the abstract-only note means the vault cannot check whether its gains, unlike the others', *require* the sensor at rollout. The most promising untried combination is SMWM's inverse-dynamics regularizer over DreamZero's video prior: DreamZero already assumes the IDM factorization but learns it end-to-end with no anti-collapse guarantee, and SMWM shows that exact objective concentrates variance on controllable DoF. Tong's scaling asymmetry is meanwhile the constraint everyone else ignores — at 51× more vision data than language at 1T params, "just add a modality" carries a bill nobody here has priced.

## Gaps

- [[papers/higuera-2026-visuo-tactile-world-models]] — `_vault note too thin — deep-read needed_`. Abstract-only: fusion mechanism, tactile hardware, baseline identity, and graceful degradation without touch are all unknown. The vault's only true extra-sensor paper is also its weakest note.
- `_needs note_`: **V-JEPA 2** (the most-cited absent paper here — Karcini's clearest JEPA↔embodied link, SMWM's anti-collapse comparator); **LAPA / UniVLA** (cross-embodiment latent actions, closest realization of Karcini's retargeting pillar); **DynaLang** (shares one representation between vision and language *inside* the world model — GenRL's direct architectural contrast); **Aether / TesserAct** (joint RGB-depth models RoboScape argues against but could not measure).
- No paper here treats audio, force/torque, or proprioception as a first-class modality; the "sensory" side of this topic is one thin note deep.

## Sources

**Core** — [[papers/higuera-2026-visuo-tactile-world-models]] (sensory fusion; thin), [[papers/shang-2025-roboscape]] (derived channels), [[papers/ivashkov-2026-sensorimotor-world-models]] (action as regularizer), [[papers/ye-2026-world-action-models]] (joint video–action denoising), [[papers/mazzaglia-2024-genrl]] (frozen VLM ↔ WM latent), [[papers/zhou-2024-robodreamer]] (compositional conditions).
**Contrast** — [[papers/tong-2026-beyond-language-modeling]] (emergence + scaling asymmetry), [[papers/zuo-2026-qwen-agentworld]] (text-only limit case).
**Framing / supporting** — [[papers/karcini-2026-robots-beyond-vla]], [[papers/ding-2024-world-models-survey]], [[papers/yao-2026-apple-pi]].
**MOCs** — [[topics/world-models]], [[topics/robotics]], [[topics/representation-learning]].
