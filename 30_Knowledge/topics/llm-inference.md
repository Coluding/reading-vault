---
type: moc
topic: llm-inference
last_updated: 2026-07-27
---

# LLM Inference

Serving trained language models fast and cheaply — as distinct from training them, and from GPU kernel work in general (see [[topics/gpu-optimization]]). The unifying constraint across these notes: **autoregressive decode is memory-bound and capacity-limited, not compute-limited.** Each generated token requires a full forward pass that mostly reads state from HBM rather than doing arithmetic, so throughput is governed by how well the system *allocates its scarce resources* — KV-cache memory and target-model batch capacity. The recurring winning move is the same in both cases: replace static provisioning with load-aware dynamic allocation.

## Memory allocation (KV cache)
- [[blogs/bansal-kv-cache]] — KV-cache management as the binding constraint in LLM serving; a six-era evolution (contiguous → PagedAttention → heterogeneous → distributed → unified) that mirrors OS memory management. Includes the Jenga numbers on waste from treating all layers uniformly (79.6% for Llama 3.2 11B Vision).

## Compute allocation (speculative decoding)
- [[papers/cheng-2026-dspark]] — **DSpark** (DeepSeek-AI): semi-autoregressive drafting (deep parallel backbone + rank-256 Markov transition head) plus confidence-scheduled verification that sizes each request's verification prefix against a profiled engine-capacity curve. +30.9% accepted length over Eagle3 offline; 60–85% faster per-user generation in live DeepSeek-V4 serving at matched throughput.

## Related topics
- [[topics/gpu-optimization]]
- [[topics/language-models]]
- [[topics/quantization]]
