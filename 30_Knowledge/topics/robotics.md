---
type: moc
topic: robotics
last_updated: 2026-07-23
---

# Robotics

_Vault notes touching robotics._

## Notes
- [[papers/jiang-2024-dexmimicgen]] — DexMimicGen (DexMG) is an automated data-generation system that turns a *handful* of human teleoperation demos into a large imitation-learning dataset
- [[papers/ye-2026-world-action-models]] — The paper introduces **DreamZero**, a 14B **World Action Model (WAM)** built on a pretrained image-to-video diffusion backbone (Wan2.1-I2V-14B) that j
- [[papers/jiang-2025-world4rl]] — **World4RL** is a two-stage framework that uses a **diffusion-based world model as a high-fidelity, frozen simulator** in which to refine an imitation
- [[papers/shi-2026-gpc-motor-control]] — GPC (Generative Pretrained Controllers) builds general-purpose, reusable controllers for physics-based character animation by borrowing the LLM recipe
- [[blogs/interlatent-ai-robotics]] — A first-principles tour of how modern "physical AI" robot policies are built, framed around the idea that a robot policy is just a function mapping ob
- [[papers/karcini-2026-robots-beyond-vla]] — This is a position/survey paper from Motoniq.ai (with collaborators at Stanford, ETH Zurich, IIT, TU Darmstadt, UCL) arguing that generalist robotics
- [[papers/higuera-2026-visuo-tactile-world-models]] — 
- [[papers/xue-2026-acwm-phys]] — **ACWM-Phys**: benchmark for **action-conditioned** video world models across rigid-body / kinematic / deformable / particle interactions; OoD generalization tracks task complexity (not physics category), and architecture ablations favor causal VAEs + cross-attention conditioning — useful signal for video-model-as-robot-simulator work.
- [[papers/zhou-2024-robodreamer]] — **RoboDreamer** (ICML 2024): compositional video-diffusion world model for robot planning; deployed on RLBench via inverse-dynamics action inference, beats UniPi and Hiveformer; 81.3% unseen-task success on RT-1 human eval.
- [[papers/chen-2026-actionable-simulators]] — position survey: world models for embodied use must be *actionable simulators* (causal structure, constraints, long-horizon stability), not visual engines; calls for closed-loop decision-oriented evaluation over FID/FVD.
- [[papers/ding-2024-world-models-survey]] — survey with a full section on world models for embodied/robotics intelligence (Dreamer, TD-MPC2, V-JEPA 2, WHAM slotted into its taxonomy) plus sim-to-real as a named open problem.
- [[papers/shang-2025-roboscape]] — **RoboScape**: autoregressive embodied world model injecting physics implicitly via joint depth-prediction + keypoint-consistency losses; SOTA robot video gen, near-real synthetic policy data, r=0.953 policy evaluator.
- [[papers/jiang-2026-robottt]] — **RoboTTT**: TTT layers turn GR00T N1.7 into a recurrent policy with gradient-updated fast weights — 8K-timestep context at constant cost; 79% vs 42% on real bimanual assembly; one-shot imitation from one human video; context length as a new scaling axis.
- [[papers/lu-2026-driftworld]] — **DriftWorld**: single-forward-pass action-conditioned world model via drifting (kernelized attraction–repulsion field, no iterative denoising); 17× faster than diffusion WMs at matching quality; Push-T IoU 0.635→0.781 with 50-proposal ranking; 0.95+ policy-eval correlation.

## Organizations
- [[websites/embodyx]] — Silicon Valley physical-AI startup building edge-first VLM/VLA foundation models for robots, vehicles, and cameras; founders Weiwei Chen & Yanzhi Wang (Northeastern) authored [[papers/zhao-2026-phyworld]] and [[papers/lin-2026-phyground]]. Backed by NSF, NVIDIA, Google AI for Startups.

## Syntheses

- [[_synthesis/multimodal-sensory-world-models]] — how world models add channels beyond RGB: touch, depth, action, language (11 papers, 2026-07-23)

