---
type: blog
title: "Coalesced Matrix Multiplication in CUDA"
author: "Lukas Bierling"
url: https://medium.com/@lukasbierling/coalesced-matrix-multiplication-in-cuda-1d914fb7084b
rw_id: 01ksfjbaprxe5qyjetbrvt3a80
topics: [cuda-programming, gpu-optimization]
priority: medium
read_state: queued
added: 2026-05-25
last_updated: 2026-05-25
---

# Coalesced Matrix Multiplication in CUDA

**TL;DR** — Second entry in the author's CUDA Algorithm Series. It takes the naive one-thread-per-output-element matmul kernel and shows that a *single change* — swapping which block/thread index maps to the row vs. the column of the output — turns scattered, strided global-memory loads into **coalesced** loads. On an A100 at $M{=}8192, N{=}6144, K{=}4096$ this drops runtime from **1433.36 ms → 136.04 ms** (~10.5×). The kernel stays memory-bound afterward; the post explicitly teases shared-memory tiling as the next step. Heavily credits Simon Boehm's CUDA-MMM blog.

## Context

By **Lukas Bierling** (the vault owner), Medium, 2026-01-21. It assumes the CUDA hierarchy (threads → block → grid, ≤1024 threads/block, blocks across SMs) and frames matmul as "the most fundamental algorithm of modern AI" since every neural-net forward pass is dominated by it. Explicitly derived from / crediting [Simon Boehm's "How to Optimize a CUDA Matmul Kernel"](https://siboehm.com/articles/22/CUDA-MMM).

## Core argument

**The naive kernel.** For $C = AB$ with $A \in \mathbb{R}^{M\times N}$, $B \in \mathbb{R}^{N \times K}$, assign one thread per output element and have it compute the dot product of a row of $A$ and a column of $B$:

$$C_{r,c} = \sum_{i=0}^{N-1} A_{r,i} \cdot B_{i,c}$$

Inspecting the PTX inside the inner loop shows **two global loads per iteration** (8 bytes) for **two FLOPs** → arithmetic intensity $2/8 = 0.25$ FLOP/byte. The kernel is firmly **memory-bound**: global-memory bandwidth, not compute, is the ceiling.

**Why the naive kernel wastes bandwidth.** The GPU services global memory in fixed cache-line transactions (32/64/128 bytes) and tries to *coalesce* a warp's 32 requests into as few transactions as possible. Coalescing works only when the 32 threads of a warp hit contiguous addresses. In the naive kernel (`c = blockIdx.y*blockDim.y + threadIdx.y; r = blockIdx.x*blockDim.x + threadIdx.x;`), matrices are row-major (`A[i,j] = A[i*K + j]`) and the first 32 threads of a warp are merged row-wise in the x-dimension. With that mapping, consecutive threads load `A[0,0], A[1,0], …, A[M,0]` — addresses **strided by $K$**, so the warp issues up to $M$ separate transactions instead of one, burning bandwidth.

**The fix — coalesced access.** Reverse the mapping so the block's x-dimension indexes the *column* of $C$ and the y-dimension indexes the *row*:

```
// naive:   r = blockIdx.x*blockDim.x + threadIdx.x;  c = blockIdx.y*...y
// coalesced: r = blockIdx.y*blockDim.y + threadIdx.y;  c = blockIdx.x*...x
```

(plus the matching swap of the two args to `blocksPerGrid`). Now consecutive threads in a warp load `B[0,1], B[0,2], …` — contiguous in row-major layout — while the `A` operand becomes a per-warp broadcast (all threads read the same `A[0,0]`). One operand is a broadcast, the other is fully coalesced; the warp collapses many transactions into one. **Everything else in the kernel stays identical.**

## Notable details

- **The whole speedup is a thread-index remap.** No tiling, no shared memory, no new algorithm — just making the warp's access pattern contiguous in `B`. ~10.5× on an A100 (1433.36 ms → 136.04 ms) at $M{=}8192, N{=}6144, K{=}4096$.
- **Coalescing ≠ data reuse.** The post is careful: the coalesced kernel is *still* memory-bound. Each output element still re-reads its row/column from global memory in the inner loop; there's no reuse beyond what the hardware does implicitly at the warp level. Arithmetic intensity is essentially unchanged — you've stopped *wasting* bandwidth, not stopped *needing* it.
- **Row-major matters.** The entire argument rests on C/C++ row-major storage; the "which index is contiguous" reasoning would flip for column-major.
- **Next step teased**: block-tiled matmul — load tiles of $A$ and $B$ into shared memory and reuse them across many multiply-adds to raise arithmetic intensity and approach peak FLOPS. (Companion to the shared-memory cooperation pattern in [[blogs/bierling-prefix-scan]].)

## Why it matters [analyst's view]

This is the cleanest possible demonstration of the single most important GPU-performance lesson: **for memory-bound kernels, the access *pattern* dominates the arithmetic.** A one-line index swap buying 10× is a visceral anchor for "coalescing is not a micro-optimization." It's the concrete worked example behind concept #11 (memory coalescing) and #10 (global memory is slow) in [[blogs/aiwithmaha-cuda-concepts]], and it sets up the standard optimization ladder (naive → coalesced → shared-memory tiled → register-tiled → warptiling) that Boehm's reference walks end-to-end. For the vault's broader AI-systems thread, this is *why* matmul-heavy training/inference is bandwidth-limited and why so much kernel engineering (tiling, fusion, flash-attention) is fundamentally about touching HBM less — the same pressure that drives KV-cache memory management in [[blogs/bansal-kv-cache]].

## Connections

- Topic MOCs: [[topics/cuda-programming]], [[topics/gpu-optimization]]
- Same series: [[blogs/bierling-prefix-scan]] (shared-memory cooperation; tiling is the named next step)
- Foundational concepts: [[blogs/aiwithmaha-cuda-concepts]] (#11 coalescing, #12 warps, #10 global memory)
- Related (downstream pressure): [[blogs/bansal-kv-cache]] (memory-bound decode), [[blogs/pytorch-torch-amp]] (matmul is the op AMP routes to low precision)
- External reference: Simon Boehm, "How to Optimize a CUDA Matmul Kernel" (siboehm.com/articles/22/CUDA-MMM)
- Author index: [[authors/lukas-bierling]]

## Selected quotes

> "Our arithmetic intensity inside the loop is 2/8 = 0.25 (FLOPs/Bytes), i.e. the performance of our kernel is memory-bound."

> "On the same problem size, we get a runtime of 136.04 ms on the same hardware!" (vs. 1433.36 ms naive)

> "Despite this improvement, the kernel remains fundamentally memory-bound ... no explicit data reuse across iterations is exploited beyond what the hardware provides implicitly at the warp level."
