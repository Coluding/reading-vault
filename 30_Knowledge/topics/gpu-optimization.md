---
type: moc
topic: gpu-optimization
last_updated: 2026-07-27
---

# GPU Optimization

Making GPU workloads fast and memory-efficient, as distinct from merely *writing* a working kernel (see [[topics/cuda-programming]]). The recurring theme across these notes: modern GPU work — training and especially LLM inference — is **memory-bound**, so the wins come from touching HBM less and packing it tighter: coalesced access patterns, low-precision dtypes routed to tensor cores, and smarter caching/paging of inference state.

## Memory access & bandwidth
- [[blogs/bierling-coalesced-matmul]] — the canonical lesson that access *pattern* dominates arithmetic for memory-bound kernels: a one-line thread-index swap → coalesced loads → ~10.5× (1433 ms → 136 ms, A100). Kernel stays memory-bound; tiling teased next.
- [[blogs/aiwithmaha-cuda-concepts]] — concept-level levers: coalescing (#11), occupancy (#13), the host↔device memory-transfer bottleneck (#14), warp divergence (#12).

## Precision
- [[blogs/pytorch-torch-amp]] — PyTorch automatic mixed precision: `autocast` per-op dtype selection + `GradScaler` loss scaling; routes matmul/conv to fp16/bf16 (tensor cores) while keeping reductions/losses in fp32.

## Inference-time memory (KV cache)
- [[blogs/bansal-kv-cache]] — KV-cache management as the binding constraint in LLM serving; six-era evolution (contiguous → PagedAttention → heterogeneous → distributed → unified) mirroring OS memory management.

## Inference-time compute scheduling
- [[papers/cheng-2026-dspark]] — **DSpark** (DeepSeek-AI): speculative decoding where verification length is a *scheduling* decision against a profiled `SPS(batch)` engine-capacity table, not a fixed hyperparameter. Production notes are the draw: O(V)→O(d) hidden-state communication in training, ZOS/CUDA-graph-compatible async scheduling, variable-length decode kernels via a marker tensor.

## Efficient architectures

- [[papers/zhu-2026-sana-wm]] — **SANA-WM** (NVIDIA): 2.6B camera-controlled world model generating 60 s 720p video on a single GPU via frame-wise Gated DeltaNet + sparse-softmax hybrid DiT; beats larger open baselines on action-following at ~36× throughput.

## Related topics
- [[topics/cuda-programming]]
- [[topics/llm-inference]]
- [[topics/mixed-precision]]
