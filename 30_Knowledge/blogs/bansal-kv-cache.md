---
type: blog
title: "KV Cache Explained: The Complete Guide to KV Cache in LLM Inference"
author: "Luv Bansal"
url: https://medium.com/@luv-bansal/the-evolution-of-kv-cache-from-simple-buffers-to-distributed-memory-systems-df51cb8ce26f
rw_id: 01ksfjchanar9n1kamphx9cbrx
topics: [llm-inference, gpu-optimization]
priority: high
read_state: queued
added: 2026-05-25
last_updated: 2026-05-25
---

# KV Cache Explained: The Complete Guide to KV Cache in LLM Inference

**TL;DR** — A practitioner survey framing KV-cache management as a six-era evolution that mirrors OS memory management (contiguous allocation → paging → distributed shared memory), compressed into ~8 years. The throughline: for production LLM serving, **KV cache, not model weights, is the binding memory constraint**, and the field's big leaps (PagedAttention, disaggregated inference, hierarchical/unified caches) are all about packing it more efficiently. Includes a concrete memory-math example, per-era pros/cons, and framework-level comparisons (vLLM vs SGLang vs TensorRT-LLM). Explicitly credited as inspired by Brian Zhang's "Five Eras of KVCache."

## Context

By **Luv Bansal** on Medium (2026-02-21), drawing on "deploying 250+ models." It opens by separating LLM inference into two phases — **prefill** (process all input tokens in parallel; compute-bound) and **decode** (generate tokens autoregressively, one at a time; memory-bound because the GPU mostly reads KV cache from HBM rather than doing math). The KV cache stores per-token Key/Value vectors so each new token can attend to all prior tokens without recomputation.

The motivating number: for **Llama-3-70B at 8K context**, KV cache per token is $2(\text{K{+}V}) \times 80\ \text{layers} \times 8\ \text{KV heads} \times 128\ \text{head\_dim} \times 2\ \text{bytes} \approx 320\ \text{KB}$. So 8K tokens ≈ 2.56 GB/request, and 32 concurrent requests ≈ **81.9 GB — more than an entire A100 80GB**, with zero room left for weights. That's the whole reason the field exists.

## Core argument

The six eras (Era 0–5):

- **Era 0 — Pre-GenAI (<2017).** Stateless feed-forward nets (ResNet, YOLO, VGG); no KV cache concept. ONNX Runtime / TensorRT built for load-forward-return.
- **Era 1 — Continuous KV cache (2017).** With the Transformer, naive engines (early HF Transformers) pre-allocate a *contiguous* `max_seq_len` block per request. Simple and a huge speedup over recompute, but memory scales with `max_seq_len × batch_size` not actual length → massive internal fragmentation. Profiling showed **only 20–38% of allocated KV memory held useful state**.
- **Era 2 — PagedAttention (2023).** vLLM (UC Berkeley) borrows OS **virtual memory + paging**: split the cache into fixed-size pages mapped by a block table, allocated on demand. Results: **2–4× throughput** over FasterTransformer/Orca, **<4% fragmentation** (vs 60–80%), hundreds–thousands of concurrent requests, and prefix sharing (SGLang's RadixAttention) for shared system prompts / RAG. Now the de-facto standard across vLLM, SGLang, TensorRT-LLM. Cost: non-contiguous attention kernels, block-size tuning, and it assumes *homogeneous* KV (same size every layer).
- **Era 3 — Heterogeneous KV caches (2024).** The "KV cache" abstraction stretches: speculative decoding (separate draft + target caches), VLM vision embeddings, quantized (FP8) KV with scaling factors, sliding-window attention, Mamba/SSM recurrent state, and **hybrid models** mixing layer types (SWA+full in Gemma 2/3; Mamba+full in Jamba/Bamba; chunked+full in Llama 4). The **Jenga paper (arXiv:2503.18292)** quantifies the waste from treating all layers uniformly: **79.6% for Llama 3.2 11B Vision**, 25% for Gemma-2, 56.25% for Ministral. Frameworks ended up with separate per-type managers — fragile.
- **Era 4 — Distributed KV cache (2025+).** KV management becomes datacenter-scale. **DistServe** disaggregates prefill (compute-bound) from decode (memory-bound) onto different instances → **4.48× more requests** (or 10.2× tighter SLO); challenge is shipping KV from prefill to decode nodes. **NVIDIA Dynamo** adds KV-cache-aware routing (route to the instance already holding the relevant cache). **Mooncake** (Moonshot AI) does hierarchical caching — spill cold pages GPU HBM → CPU DRAM → SSD, overlapping load/store with the prior layer's compute → up to **525% throughput** in long-context, **75% more requests** serving Kimi.
- **Era 5 — Unified hybrid KV caches (2025+).** The frontier: heterogeneous KV types share one memory pool (composability). **Jenga** uses the **LCM of embedding sizes** as a huge-page size (e.g. `LCM(256, 384) = 768`) subdivided into type-specific small pages → up to **79.6% better GPU memory utilization, 4.92× throughput (1.80× avg)** over vanilla vLLM. **SGLang** uses CUDA Virtual Memory APIs for elastic, resizable pools (e.g. dynamically rebalancing Mamba-pool vs KV-pool).

## Notable details

- **Decode is memory-bound, prefill is compute-bound** — the single fact that explains disaggregation and why decode dominates serving cost.
- **vLLM vs SGLang prefix caching**: vLLM uses hash-based block-level matching; SGLang uses a RadixAttention LRU tree. Author finds SGLang gets higher hit rates on complex multi-call workloads (agents, tree-of-thought); vLLM is simpler and fine for standard chat.
- **Practitioner routing guide**: standard text serving → Era 2 (PagedAttention, vLLM/SGLang + prefix caching); VLMs → Era 3 + vLLM encoder disaggregation; hybrid architectures → Era 5 (SGLang CUDA-VM / Jenga LCM); high-throughput at scale → Era 4 (disaggregation + KV-aware routing, Dynamo/Mooncake); 100K+ long context → Era 4 hierarchical spillover (mandatory or you OOM).
- **Emerging infra**: NVIDIA Dynamo, vLLM Production Stack, llm-d, AIBrix (Kubernetes-native).

## Why it matters [analyst's view]

This is the best single-document mental model I've seen for *why LLM serving is hard*, and it slots directly above the low-level GPU notes in this vault. The coalesced-matmul note ([[blogs/bierling-coalesced-matmul]]) establishes that GPU work is bandwidth-bound at the kernel level; this note shows the *same pressure* one level up — decode is memory-bound, so the entire serving stack is an exercise in touching HBM less and packing the cache tighter. The OS-analogy framing (contiguous → paging → virtual memory → distributed shared memory) is the exportable insight and makes the unfamiliar (RadixAttention, Mooncake hierarchies) legible via the familiar. Caveats: it's a vendor-comparison survey leaning on others' headline numbers (Jenga, DistServe, Mooncake papers), so treat the specific multipliers as *reported* rather than independently verified — the arXiv:2503.18292 Jenga paper is the obvious one to chase for the heterogeneity claims.

## Connections

- Topic MOCs: [[topics/llm-inference]], [[topics/gpu-optimization]]
- Related (low-level cause): [[blogs/bierling-coalesced-matmul]] (kernels are bandwidth-bound — same pressure, one level down)
- Related (precision lever): [[blogs/pytorch-torch-amp]] (FP8/FP16 KV quantization is the same low-precision tradeoff)
- Foundational concepts: [[blogs/aiwithmaha-cuda-concepts]] (#8–10 device/global memory, #14 memory-transfer bottleneck)
- Papers to chase: Jenga (arXiv:2503.18292), vLLM/PagedAttention, DistServe, Mooncake — none yet in vault
- Author index: [[authors/luv-bansal]]

## Selected quotes

> "the model weights are only half the story. The other half is KV cache ... Managing this memory efficiently is what separates a sluggish demo from a production-grade inference system."

> "[Llama-3-70B, 8K context, 32 concurrent requests] = 81.9 GB. That is more than an entire A100 80GB GPU just for KV cache, leaving zero room for model weights."

> "The evolution of KV cache management mirrors the evolution of operating system memory management ... The difference is that this evolution happened in just 8 years instead of 40."
