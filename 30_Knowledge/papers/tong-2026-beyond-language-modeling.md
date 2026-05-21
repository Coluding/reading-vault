---
type: paper
title: "Beyond Language Modeling: An Exploration of Multimodal Pretraining"
authors: ["Shengbang Tong", "David Fan", "John Nguyen", "Ellis Brown", "Gaoyue Zhou", "Shengyi Qian", "Boyang Zheng", "Théophane Vallaeys", "Junlin Han", "Rob Fergus", "Naila Murray", "Marjan Ghazvininejad", "Mike Lewis", "Nicolas Ballas", "Amir Bar", "Michael Rabbat", "Jakob Verbeek", "Luke Zettlemoyer", "Koustuv Sinha", "Yann LeCun", "Saining Xie"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2603.03276
rw_id: 01ks5mjw1pv57jart7tvrccrb0
topics: [multimodal-pretraining, mixture-of-experts, scaling-laws, vision-language, world-models]
priority: high
read_state: queued
relevance: ""
added: 2026-05-21
last_updated: 2026-05-21
---

# Beyond Language Modeling: An Exploration of Multimodal Pretraining

**TL;DR** — A large-scale, from-scratch, controlled study of native multimodal pretraining (text + image + video + action-conditioned video) using the Transfusion framework (next-token-prediction for language, diffusion for vision). Four reported takeaways: (i) the Representation Autoencoder (RAE) is the right unified visual representation; (ii) vision and language data are complementary; (iii) unified multimodal pretraining produces emergent world-modeling capabilities; (iv) Mixture-of-Experts is the architectural fix for the *scaling asymmetry* the paper uncovers — vision is significantly more data-hungry than language, and MoE lets you give language enough capacity while feeding vision enough data.

## Context & motivation

[from the abstract]

- "The visual world offers a critical axis for advancing foundation models beyond language. Despite growing interest in this direction, the design space for native multimodal models remains opaque."
- The paper's framing problem: prior multimodal work piggybacks on language pretraining, so the *intrinsic* dynamics of multimodal pretraining are hard to isolate. The contribution is therefore methodological — "controlled, from-scratch pretraining experiments, isolating the factors that govern multimodal pretraining without interference from language pretraining."
- Subject area filed under cs.CV.

Specific prior-work baselines, exact dataset names, and benchmark suites are _not addressed by the source_ at the abstract level.

## Method

### Framework
- "We adopt the Transfusion framework, using next-token prediction for language and diffusion for vision" (abstract). This is the unified-objective scaffold for the entire study — language tokens trained autoregressively, vision tokens trained with a diffusion objective, in a single model.

### Training data
- Mix of "text, video, image-text pairs, and even action-conditioned video" (abstract).
- Specific dataset identities, sizes, and curation procedures are _not addressed by the source_.

### Architectural choices being studied
- **Visual tokenizer / unified visual representation** — multiple candidates compared; the headline finding is that **Representation Autoencoder (RAE)** wins on jointly excelling at *understanding* and *generation*.
- **Mixture-of-Experts (MoE)** — the architectural lever that lets the model spend capacity asymmetrically across modalities.
- Layer counts, expert counts, routing scheme, optimizer, and schedule are _not addressed by the source_.

### Scaling-law experimentation
- "Through IsoFLOP analysis, we compute scaling laws for both modalities and uncover a scaling asymmetry: vision is significantly more data-hungry than language" (abstract).
- Exact exponents and IsoFLOP curves are _not addressed by the source_.

## Experimental setup

_Not addressed by the source_ at the abstract level. The paper is positioned as a controlled IsoFLOP study, so a full setup section presumably exists in the body.

## Key results

[from the abstract, only — no specific numbers given]

- **(i) RAE is the right unified visual encoder** — "provides an optimal unified visual representation by excelling at both visual understanding and generation."
- **(ii) Vision + language complementarity** — "visual and language data are complementary and yield synergy for downstream capabilities."
- **(iii) Emergent world modeling** — "unified multimodal pretraining leads naturally to world modeling, with capabilities emerging from general training."
- **(iv) MoE harmonizes the asymmetry** — "Mixture-of-Experts (MoE) enables efficient and effective multimodal scaling while naturally inducing modality specialization."
- **Scaling-law headline** — "vision is significantly more data-hungry than language."

Specific benchmark scores, win-rates, and synergy magnitudes are _not addressed by the source_.

## Ablations

_Not addressed by the source_ at the abstract level. The four-insight structure implies a substantial ablation program in the body.

## Limitations

_Not addressed by the source_ at the abstract level.

[analyst's view] Reading-between-the-lines flags worth checking against the body:
- Is the scaling asymmetry an artifact of *current* visual data quality (web-scraped vs. captioned) rather than an intrinsic property of the modality?
- How sensitive is the RAE conclusion to the diffusion objective specifically — would discrete-token vision change the verdict?

## Why it matters [analyst's view]

Three things land:

1. **The "scaling asymmetry" framing is the headline-worthy finding.** If vision is genuinely more data-hungry than language at matched FLOPs, every native-multimodal scaling plan needs to budget data and capacity by modality rather than by parameter count. This is a different mental model than the dominant "one Chinchilla curve for all tokens" view.
2. **MoE as a *physical* answer to *statistical* asymmetry.** Routing different modalities to different experts is not just compute-efficiency — the paper frames it as the architecture that lets language take the capacity it needs while vision soaks up the data it needs. That converts MoE from a scaling trick into a multimodal *structural* prior.
3. **Emergent world modeling from generic pretraining** is the bridge to the JEPA / world-model line. The paper does *not* train with a world-modeling objective explicitly — it claims the capability "emerges from general training" on action-conditioned video. That's a strong claim if true, and a direct contrast with [[papers/maes-2026-leworldmodel]] which builds an explicit world model with two simple losses.

The bigger-picture takeaway: the team — FAIR/Meta + NYU — is staking out a position that "the right way to learn world models is to scale unified multimodal pretraining," not to invent task-specific objectives.

## Open questions

[analyst's view]
- What is *RAE* exactly here? The abstract names it as a discovery, but Representation Autoencoder is not yet a vault concept. (Worth a follow-up read of the body or a citation chase.)
- How does the IsoFLOP scaling law compare quantitatively against Chinchilla? The "vision is more data-hungry" claim only bites if the exponent gap is large.
- Does the "emergent world modeling" claim survive contact with the [[papers/joseph-2026-physics-video-world-models]] interpretability result that physics is encoded in distributed, non-factorised representations?
- Does MoE-induced "modality specialization" mean experts cleanly partition by modality, or is there cross-modal mixing? Different stories about what MoE is doing under the hood.

## Connections

- Topic MOCs: [[topics/world-models]]
- Related papers:
  - [[papers/maes-2026-leworldmodel]] — explicit JEPA-based world model from pixels; a structural alternative to Tong et al.'s "emerges from scale" framing.
  - [[papers/joseph-2026-physics-video-world-models]] — interpretability of physics-related capabilities in video encoders; relevant to the "emergent world modeling" claim.
- Author indices: [[authors/shengbang-tong]]

## Selected quotes

> "We provide empirical clarity through controlled, from-scratch pretraining experiments, isolating the factors that govern multimodal pretraining without interference from language pretraining." — abstract

> "Through IsoFLOP analysis, we compute scaling laws for both modalities and uncover a scaling asymmetry: vision is significantly more data-hungry than language." — abstract

> "The MoE architecture harmonizes this scaling asymmetry by providing the high model capacity required by language while accommodating the data-intensive nature of vision, paving the way for truly unified multimodal models." — abstract
