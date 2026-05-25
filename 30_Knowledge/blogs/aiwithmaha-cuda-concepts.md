---
type: blog
title: "20 CUDA Concepts Explained Simply"
author: "AIwithMaha"
url: https://medium.com/@mahareddyroja247/20-cuda-concepts-explained-simply-b23b89918ba0
rw_id: 01ksfjdebzdh329s7t4sjxkxr0
topics: [cuda-programming, gpu-optimization]
priority: low
read_state: queued
added: 2026-05-25
last_updated: 2026-05-25
---

# 20 CUDA Concepts Explained Simply

**TL;DR** — A beginner-friendly glossary that demystifies "what actually happens when you call `model.to('cuda')`" through 20 short, analogy-driven concepts: the execution hierarchy (kernel → grid → blocks → threads → warps), the memory hierarchy (global vs shared vs device memory, coalescing), and the performance levers (occupancy, memory-transfer bottleneck, streams, tensor cores, cuDNN). No code depth — it's a conceptual map for PyTorch/TensorFlow users who want to read GPU error messages and performance issues with understanding rather than write raw kernels.

## Context

By **AIwithMaha** (Maha Reddy Roja) on Medium, 2026-04-01. Audience: ML practitioners who use GPUs through frameworks but never learned what the GPU is doing underneath. The author's hook: "I used CUDA for a while before I properly understood what was happening behind the scenes. Tutorials usually tell you what to do, but not what the GPU is actually doing." Each concept gets a plain-language analogy (workers, factory lines, whiteboards, neighborhoods).

## Core argument

The 20 concepts, grouped:

**What/why (1–2):** CUDA = Compute Unified Device Architecture, NVIDIA's way to run code on the GPU (NVIDIA-only). CPU = a few smart cores (8–32) good at branching/decisions; GPU = thousands of simple cores doing the same thing at once.

**Execution hierarchy (3–7):** A **kernel** is a GPU function (`__global__`); in PyTorch every tensor op launches kernels under the hood. A kernel runs as thousands of **threads** (one small piece of work each). Threads group into **blocks** (threads in a block can communicate; across blocks they can't). The **grid** is all the blocks — structure is grid → blocks → threads ("city → neighborhoods → houses"). Each thread uses its **thread ID** (block number + position) to pick which data element it owns.

**Memory hierarchy (8–11):** **Device/global memory** is the GPU's own RAM, separate from host RAM; data must be copied host→device before work. **Shared memory** is a small, very fast per-block scratchpad ("a whiteboard in the room") — far faster than global. **Global memory** is big but slow; good CUDA minimizes touching it. **Memory coalescing**: if threads read sequential addresses the GPU fetches them in one chunk (fast); scattered reads → many trips (slow).

**Performance model (12–17):** A **warp** is exactly 32 threads executing the same instruction in lockstep — which is why branching (if/else) is expensive (divergent warps run both paths). **Occupancy** = how busy the GPU is (more concurrent threads → usually faster, up to a point). **Memory-transfer bottleneck**: host↔device copies are slow relative to compute; a kernel that runs in 1 ms but needs 10 ms of transfer is a net slowdown — keep tensors on the GPU, avoid accidental `.cpu()`. **Parallelism vs concurrency**: concurrency = taking turns; parallelism = literally simultaneous (what GPUs do). **CUDA streams** overlap copy/compute/copy-back to cut idle time. **Synchronization** (`torch.cuda.synchronize()`): the CPU launches kernels asynchronously and only waits when it needs results.

**Hardware/library payoff (18–20):** **FLOPS** = floating-point ops/sec (modern GPUs do hundreds of teraFLOPS). **Tensor cores** (newer NVIDIA GPUs) do a small matrix multiply in one operation — huge for neural nets, and the reason mixed precision (fp16) is used: to route ops to tensor cores. **cuDNN** is NVIDIA's library of pre-optimized DL ops (conv, activation, pooling); most of PyTorch's speed comes from cuDNN, not raw CUDA — "the speed isn't magic, it's accumulated engineering."

## Notable details

- **The framework connection is the payoff**: `z = x + y` on CUDA tensors silently launches thousands of threads; convolutions call cuDNN; `torch.cuda.amp` routes ops to tensor cores; vectorized ops beat Python loops partly because they produce uniform warp work.
- **Warp divergence** explained as the concrete cost of branching — a useful intuition for why GPU code avoids data-dependent control flow.
- **Most common real-world bug named**: mixing CPU and CUDA tensors / accidental `.cpu()`, where transfer time dwarfs kernel time.

## Why it matters [analyst's view]

This is the **glue note** for the GPU cluster in the vault — the shared vocabulary the deeper notes assume. It's deliberately shallow (no code, all analogy), so its value is as an index/onboarding map rather than a reference: when [[blogs/bierling-coalesced-matmul]] talks about warps issuing strided transactions, or [[blogs/bierling-prefix-scan]] leans on shared memory + `__syncthreads`, or [[blogs/bansal-kv-cache]] calls decode "memory-bound," *this* note is where each of those terms gets its plain-language definition. Concepts #19 (tensor cores) and #11 (coalescing) are the two that the other notes turn into worked examples ([[blogs/pytorch-torch-amp]] and [[blogs/bierling-coalesced-matmul]] respectively). Low priority precisely because it's introductory — keep it as the entry point of the [[topics/cuda-programming]] MOC, not as something to re-read.

## Connections

- Topic MOCs: [[topics/cuda-programming]], [[topics/gpu-optimization]]
- Worked examples of these concepts: [[blogs/bierling-coalesced-matmul]] (#11 coalescing, #12 warps), [[blogs/bierling-prefix-scan]] (#9 shared memory, #5 blocks), [[blogs/pytorch-torch-amp]] (#19 tensor cores, mixed precision), [[blogs/bansal-kv-cache]] (#14 memory-transfer / memory-bound)
- Author index: [[authors/aiwithmaha]]
