---
type: blog
title: "World models and interpretability are two sides of the same coin"
author: "Sonia Joseph"
url: https://www.soniajoseph.ai/world-models-and-interpretability-are-two-sides-of-the-same-coin-2/
rw_id: 01kvq3pj6w59j4bc0t04z1ymwm
topics: [world-models, interpretability, mechanistic-interpretability, representation-learning]
priority: high
read_state: queued
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

Sonia Joseph proposes a fourth definition of "world model" that cuts against the dominant generative/3D/latent-prediction framings: the **Internal World Model** — not a separate module and not the model's *output*, but "the internal state variables and dynamics learned by the foundation model itself," i.e. a foundation model's internal causal structure. Her central claim is that **world models and interpretability are two sides of the same coin**: both fields are really asking whether a model has learned latent variables corresponding to meaningful structure in the world (velocity, object permanence, causality) and how those variables interact to produce predictions. She recasts interpretability as two more precise disciplines — **causal discovery** (identify the latent variables that causally generate a prediction, exactly the question climate scientists and cosmologists have long asked) and **white-box evaluation** (inspect internal representations rather than only outputs, e.g. detecting deception or a robot about to drop a container). Worth remembering as the sharpest statement of the "world model = internal mechanics, not generated video" position, and for its concrete claim that if we can *steer* internal variables and observe counterfactual changes, foundation models start to look less like predictors and more like **simulators with unsupervised-learned physics "dials."**

## Context

A short (~1.5k word) research-notebook essay (June 2026) on Sonia Joseph's personal site. Joseph works on mechanistic interpretability and physical reasoning via JEPA models on Newtonian dynamics; this post sits alongside her empirical work (in the vault, [[papers/joseph-2026-physics-video-world-models]] on how video models encode physics). It responds directly to the definitional confusion in the field — "depending on who you ask, a world model is a generative model, a 3D reconstruction model, or a latent-space prediction model" — and proposes returning to "the original," the Internal World Model of cognitive science. The framing crystallized through conversations with causality-in-climate researcher Christina Last: they realized they were "asking the same question using different scientific languages" (mech interp vs pre-deep-learning climate science), both trying to discover the latent variables generating a physical system. The piece is explicitly frontier-speculative: "we are very much operating at the frontier," and she flags that roughly half the experts she talks to lean each way on the key open question.

## Core argument

**1. Reframe the object of study.** Instead of asking whether a model can generate realistic videos or predict future latent states, ask whether it has learned *variables that correspond to meaningful structure in the world* and how those variables interact to produce predictions and actions. The object of study is no longer generation or latent prediction — it is **the model's internal mechanics**.

**2. Interpretability = causal discovery + white-box evaluation.**
- *Causal discovery*: identifying the latent variables that causally generate a prediction. This is "immediately familiar to scientists of the physical world" — climate researchers search for the variables generating typhoons, cosmologists for the variables behind star formation. These communities did causal-discovery-esque work for decades before deep learning.
- *White-box evaluation*: inspect the internal representations that produced an output, not just the output. In LLMs: internal signatures of deception, hallucination, or planning. In embodied systems: dangerous features like a child interacting with an electrical outlet, or a robot arm about to drop a container of liquid. The shift is from "what did the model predict" to "how did it arrive there" — accessing the model's **internal reasoning trace, the chain of latent variables** that combined to produce the next action/prediction/frame.

Both disciplines are fundamentally concerned with identifying the variables that causally generate a prediction — hence "two sides of the same coin."

**3. From predictors to simulators.** The provocative move: what if foundation models are not merely predictors but *simulators*? She is careful to say she does **not** mean simulator in the generative sense (Sora, Genie, World Labs) *or* even the LeCun/JEPA latent-space sense — those are about the reconstruction target (pixels vs latents), which she treats as an efficiency question, not a different object of study. She means the *internal dynamics* begin to resemble a simulator: if a video model develops an internal representation of velocity, and we can **steer** that velocity representation and observe **corresponding counterfactual changes** in future predictions, then its latent variables correspond to physical quantities in a factorized way — like the "dials" of a physics simulator, except the dials were learned unsupervised from raw observation.

**4. Cognitive maps — the open empirical question.** Do large foundation models actually develop cognitive maps (an internal model of the environment), or do they succeed via stimulus-response? Spectrum: a jellyfish (pure stimulus-response, no cognitive map of the ocean) vs a person who can close their eyes and mentally simulate walking through a familiar bedroom, running counterfactuals (knocking over a laundry basket, opening a drawer) entirely in their head. Evidence for emergence: the famous transformer trained on New York taxi trajectories develops "a distorted but recognizable cognitive map of the city" (cites Teoh et al. 2026). But the jury is genuinely out — Joseph notes some of her own JEPA findings "point against the strongest versions of this hypothesis."

## Notable details

- **The efficiency caveat on latent prediction:** latent prediction "may produce more factorized variables, because a model not forced to reconstruct irrelevant detail can spend its capacity on the relevant causal variables instead. But that's an efficiency advantage, not a different object of study." — a pointed reframing that subordinates the generative-vs-JEPA debate to her internal-mechanics lens.
- **Physical verification (application 1, via Dileep George's flight-simulator example):** a pilot's brain may contain all the right representations of altitude, velocity, instrument readings — traditional interpretability can confirm those representations *exist* — yet the pilot can still crash if the representations "fail to combine correctly in a particular situation." The AI analog: a model may contain the right knowledge yet fail to *recruit* it correctly. "Understanding the causal interactions between internal variables in specific contexts is far beyond simply locating a representation with a linear probe." — a direct critique of probe-based interpretability.
- **Scientific simulation (application 2):** unsupervised foundation models trained on climate data (Walrus, GraphCast; work from Theodore MacMillan at Stanford, Polymathic, Google DeepMind) can recover interpretable variables like typhoons — "one can imagine steering vectors corresponding to phenomena such as typhoons." Protein-folding features (work supervised by @wendlerch and @davidbau). Extrapolations: cosmologists training on baryonic-matter evolution recovering latent variables governing star formation; in the extreme, a model discovering causal variables current theories miss, "forcing revisions to our understanding of cosmology or even general relativity."
- **Unifying claim:** "The same framework applies from flight simulators to the Big Bang." Whether verifying a robot arm, modeling a supply chain, simulating an energy grid, or studying the early universe, the technique is the same: trace how observations combine into internal variables and how those variables interact through the network's internal circuitry to generate behavior.
- **Named open questions:** Do foundation models actually develop cognitive maps or succeed via jellyfish-like strategies? What are the *scaling laws* for Internal World Models — how do compute, data, architecture, and domain expertise affect their emergence? Which domains are easy to test? "What distinguishes an Einstein-level Internal World Model from a model trained on much narrower distributions?"
- Community signal: the divide is reflected in a summer GAC workshop at Columbia on cognitive maps in world models (organized by Eivinas Butkus).

## Why it matters [analyst's view]

This is a genuinely unifying reframe: by defining the world model as a foundation model's *internal causal structure*, Joseph dissolves the taxonomic squabbles (generative vs 3D vs latent) that dominate popular treatments and relocates the whole question inside mechanistic interpretability. It's the natural theoretical counterpart to her own empirical paper [[papers/joseph-2026-physics-video-world-models]], which asks whether video models actually encode physics — this post supplies the conceptual "why that's the right question." It stands in sharp, productive contrast to [[blogs/mccormick-world-models]]: McCormick's World Model is an *external, generative, playable environment you train agents inside*; Joseph's is the *internal machinery of any foundation model*, and generation is explicitly "beside the point." The steerability criterion (steer a latent, observe counterfactual downstream changes) is a concrete, testable bridge between interpretability's activation-steering toolkit and world-modeling's simulation ambitions — arguably the most actionable idea in the piece. The main thing to hold skeptically is that the causal-discovery/white-box program is asserted as a research direction, not yet demonstrated at scale on frontier models; the cognitive-map question remains genuinely open by her own admission.

## Connections

- Topic MOCs: [[topics/world-models]], [[topics/interpretability]], [[topics/mechanistic-interpretability]], [[topics/representation-learning]]
- Related papers in vault: [[papers/joseph-2026-physics-video-world-models]] (author's own empirical study of physics encoding in video world models — the direct companion to this post), [[papers/maes-2026-leworldmodel]] (JEPA world model whose latents encode recoverable physical structure — evidence relevant to the steerable-variable claim)
- Contrasts with: [[blogs/mccormick-world-models]] (external/generative vs internal/causal-structure definition of "world model")
- Author index: [[authors/sonia-joseph]]

## Selected quotes

> "The world model is not a separate module sitting alongside a foundation model, nor is it the model's generated output. The world model is the internal state variables and dynamics learned by the foundation model itself. The Internal World Model is a foundation model's internal causal structure."

> "If we can steer that velocity representation and observe corresponding counterfactual changes in future predictions, then the model starts looking less like a statistical predictor and more like a simulator whose latent variables correspond to physical quantities in a factorized way. Think about the 'dials' of a physics simulator, but the 'dials' were learned in an unsupervised way, from raw observation."

> "We would not simply generate worlds, or even just predict them. We would write down their mechanics."
