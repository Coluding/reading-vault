---
type: moc
topic: video-generation
last_updated: 2026-07-10
---

# Video Generation

_Vault notes touching video generation._

## Notes
- [[papers/esmati-2026-invisible-hand-physics]] — This paper asks whether video diffusion models internally *encode* physical structure even when their generated outputs violate physics
- [[papers/yuan-2026-physics-alignment]] — The paper (from FAIR / Meta Superintelligence Labs, dated March 2, 2026) shows that a large share of the physics-implausibility in state-of-the-art vi
- [[papers/begiristain-2026-cronos]] — CRONOS benchmark: measures whether video generators predict physical events consistently under viewpoint / scene / appearance / object-category interventions; headline finding is severe **viewpoint** sensitivity, plus best-model event success of only 22%.
- [[papers/meng-2024-phygenbench]] — **PhyGenBench** (the seminal physics-commonsense T2V benchmark): 160 prompts / 27 laws / 4 domains + hierarchical **PhyGenEval** judge; best model 0.51, scaling & prompt-engineering insufficient.
- [[papers/gu-2025-phyworldbench]] — **PhyWorldBench** (ICLR 2026 oral): 1,050 prompts across 50 phenomena with an **anti-physics** tier and an MLLM-judge (CAP); best model (Pika 2.0) only 0.262 joint success.
- [[papers/cao-2026-judgefit]] — **JudgeFit**: physics VLM-judges are unreliable and mutually inconsistent; a per-VLM discovered taxonomy lifts human agreement ~32% across 16 VLMs.
- [[papers/xiong-2026-physalign]] — **PhysAlign**: a LoRA adapter aligning Wan2.2 I2V features to a V-JEPA2 teacher + 3D depth for physics coherence, with no visual-quality tradeoff and free inference.
- [[papers/xue-2026-acwm-phys]] — **ACWM-Phys**: benchmarks action-conditioned video world models across rigid/deformable/particle/kinematics; OoD failure tracks task complexity, not physics category.
- [[papers/cai-2026-mode-mean-seeking]] — The paper attacks the "fidelity–horizon gap" in scaling video generation from seconds to minutes: long, coherent video data is scarce, so models train
