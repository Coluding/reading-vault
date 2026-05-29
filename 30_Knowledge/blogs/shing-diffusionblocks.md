---
type: blog
title: "DiffusionBlocks: Training Neural Networks One Block at a Time"
author: "Makoto Shing"
url: https://pub.sakana.ai/diffusionblocks/
rw_id: 01ksnq9sr865sd2yew5sephwd0
topics: [generative-models, diffusion-models, memory-efficient-training, block-wise-training]
priority: high
read_state: skimmed
relevance: ""
added: 2026-05-27
last_updated: 2026-05-27
---

# DiffusionBlocks: Training Neural Networks One Block at a Time

**TL;DR** — Sakana AI's project-page summary of an ICLR 2026 paper ([arXiv:2506.14202](https://arxiv.org/abs/2506.14202), Shing & Koyama). The core problem: end-to-end backprop keeps every layer's activations in memory, so training memory grows linearly with depth — the bottleneck on who can train large models. **DiffusionBlocks** partitions a network into $B$ blocks and trains *one block at a time, independently*, cutting training memory to roughly $1/B$. The trick that makes independent training principled (rather than the usual block-wise degradation seen on image classification) is casting each block's job as a **range of a diffusion denoising process**: each block is responsible for one slice of "closeness to the target," so it has a well-defined local objective and never needs to know what the other blocks are doing. They report performance "comparable to end-to-end" across 5 architectures spanning image classification, image generation, and text generation.

## Context

This is a research-blog/project page on `pub.sakana.ai`, not the arXiv paper itself, so it's filed as a blog (per CLAUDE.md Part 13's default-to-blog rule for project-page summaries). It summarizes joint work by Makoto Shing (Sakana AI) and Masanori Koyama (University of Tokyo), presented at ICLR 2026. The framing is explicitly about **democratizing training**: frontier models need thousands of GPUs because end-to-end optimization couples all parameters in memory; DiffusionBlocks attacks the *training-memory* axis specifically. Sakana positions it alongside their broader efficiency portfolio (evolutionary model merging, knowledge distillation, inference-time memory optimization, low-cost context extension, kernel-level acceleration).

## Core argument

**The memory bottleneck is structural.** End-to-end backprop must retain all intermediate activations across the network for the backward pass, so memory scales linearly with depth — and modern Transformers scale largely by adding layers.

**Block-wise training is the obvious fix, but historically lossy.** Prior block-wise work (cited [4]–[8]) mostly targeted image classification and lagged end-to-end. Extending it to Transformers and *generative* tasks was under-explored. The closest prior work is **NoProp** [9], which already pairs a diffusion framework with block-wise training on image classification; DiffusionBlocks' stated novelty is bringing this to modern Transformer architectures and generative tasks (the post defers the detailed technical comparison to the paper).

**The diffusion reframing is the load-bearing idea.** In standard training, what each layer *does* is unspecified — layers are a black box that "somehow" maps input to output. DiffusionBlocks instead assigns each block an explicit role: the dynamics of *gradually approaching the target as you move through blocks*. This matches diffusion models' denoising dynamics. The connection is anchored on the residual-connection ↔ discretized-ODE correspondence [13][14] (the same ODE class underlying diffusion): block-wise dynamics can be cast as the reverse (denoising) process of a diffusion model, which is what licenses principled *independent* per-block training.

## Notable details

- **Three-step conversion recipe** for an existing Transformer: (1) **Partition** the $L$ layers into $B$ blocks; (2) **Assign noise ranges** — each block owns a range of "closeness to the target"; (3) **Add conditioning** so each block recognizes its assigned range. Training then samples a single random block per iteration; the others aren't computed, giving the $\approx 1/B$ memory reduction.
- **Validated across 5 architectures / 3 domains**: ViT (image classification, CIFAR-100), DiT (image generation, ImageNet 256), Masked Diffusion (text gen, text8), Autoregressive Transformer (text gen, OpenWebText), and Recurrent-depth / Looped Transformer. The post states results are "comparable to end-to-end while using a fraction of training memory" but gives **no headline metric numbers** — those are in the full paper, which I have not fetched. _Specific accuracy/PPL figures: needs verification from the paper._
- **Bonus result for recurrent-depth models.** Looped Transformers (apply the same net $K$ times) normally need backpropagation-through-time over $K$ iterations — a compute bottleneck. The DiffusionBlocks view (repeated refinement = "gradual progress toward the target") lets them replace $K$-iteration BPTT with a **single forward pass** at training time, while keeping the $K$-iteration procedure at inference.
- **They sometimes beat end-to-end, not just match it.** The post flags cases where DiffusionBlocks *outperformed* end-to-end training and hypothesizes this comes from an implicit **curriculum-learning** effect: assigning each block a graded difficulty distributes learning difficulty across the network. Formal analysis is left as future work.
- **Stated future direction with the biggest upside**: converting *pretrained* large models into block-wise-trainable form via fine-tuning — which would put large-model training/adaptation within reach of individual researchers and small labs.

## Why it matters [analyst's view]

This is a genuinely different axis from the GPU-efficiency cluster already in the vault ([[blogs/bansal-kv-cache]], the CUDA-kernel notes): those optimize *how* you execute a fixed training/inference graph; DiffusionBlocks changes the *training algorithm* so the graph never has to be resident all at once. If the pretrained-model-conversion direction works, it's a memory-side analogue to what LoRA did for parameter-side fine-tuning cost — and it would interact cleanly with the [[topics/gpu-optimization]] work, since a 1/B memory footprint changes what fits on a single device. The diffusion-as-block-roles framing is also conceptually adjacent to the score-matching / vector-field-as-design-choice theme in [[topics/generative-models]] (cf. [[papers/pao-huang-2026-flux-matching]]): both treat the *dynamics* a network implements as the object to design, not an emergent afterthought. The honest caveat: without the paper's numbers, "comparable to end-to-end" is a claim to verify — block-wise methods have a long history of looking competitive on small benchmarks and degrading at scale, and the headline value (pretrained conversion) is explicitly *not yet demonstrated*.

## Connections

- Topic MOCs: [[topics/generative-models]]
- Related work mentioned: NoProp (diffusion + block-wise on classification, [9]); residual-connections-as-ODE [13][14]
- Adjacent vault notes: [[papers/pao-huang-2026-flux-matching]] (designing generative dynamics), [[blogs/bansal-kv-cache]] (orthogonal — inference-memory axis)
- Author index: [[authors/makoto-shing]]

## Selected quotes

> "DiffusionBlocks explicitly assigns a role to each block (a group of layers) ... Each block only needs to optimize toward its own individual objective, and can be trained without depending on what the other blocks are doing."

> "Since the other blocks do not need to be computed, memory consumption is reduced to roughly 1/B."

> "We can replace K-iteration BPTT training with a single forward pass during training, significantly reducing computational cost without sacrificing performance."
