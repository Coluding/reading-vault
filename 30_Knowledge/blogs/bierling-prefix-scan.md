---
type: blog
title: "Recursive Parallel Prefix Scan Using CUDA"
author: "Lukas Bierling"
url: https://medium.com/@lukasbierling/recursive-parallel-prefix-scan-using-cuda-b8181b8527a9
rw_id: 01ksfjbkp7st6hd5v16kqhx7gm
topics: [cuda-programming, parallel-algorithms]
priority: medium
read_state: queued
added: 2026-05-25
last_updated: 2026-05-25
---

# Recursive Parallel Prefix Scan Using CUDA

**TL;DR** — A tutorial-grade walkthrough (part of the author's "CUDA Algorithm Series") of computing an inclusive prefix sum (scan) on a GPU. It starts from the obvious sequential $O(N)$ loop, shows a single-block parallel kernel that uses shared memory + a Hillis–Steele doubling pattern to reach $O(\log N)$ depth, then handles the realistic case where the array spans many blocks via a **divide-and-conquer recursion**: scan each block locally, extract per-block sums, recursively scan that block-sum array, then add each block's prefix back. The author is explicit that this prioritizes clarity over work-efficiency and is not production-grade.

## Context

Authored by **Lukas Bierling** (the vault owner) on Medium, 2026-01-10, as an educational entry in his CUDA Algorithm Series. The framing is "this is a good starting point to learn parallel programming," not a production kernel. It assumes familiarity with the CUDA execution hierarchy (threads → blocks → grid; threads in a block share fast shared memory; max 1024 threads/block; blocks distributed across SMs).

## Core argument

Prefix scan *looks* inherently sequential — each output $y_i = \sum_{k \le i} x_k$ seems to need $y_{i-1}$ first — but it parallelizes if you stop insisting on linear-time work and exploit the GPU hierarchy instead.

**Single-block kernel (base case, `BLOCK_SIZE ≥ N`).** Load the input into a shared-memory array `intermediate[]`, then loop with a stride that doubles each step (`for i = 1; i < blockDim.x; i *= 2`): each thread `tidx` adds the value `i` positions to its left. After $\lceil \log_2 N \rceil$ iterations every cell holds its prefix sum. Because additions commute, the order doesn't matter — the doubling jumps just guarantee no value is double-counted. This is the **Hillis–Steele** scan: $O(\log N)$ depth.

The correctness hinge is **three-phase synchronization** per iteration:
1. `__syncthreads()` before the loop — all threads see a fully initialized `intermediate`.
2. each thread reads `intermediate[tidx - i]` into a register, then `__syncthreads()` — guarantees all reads finish before any write.
3. each thread writes `intermediate[tidx] += val`, then `__syncthreads()` — all writes visible before the next iteration.

Without the middle barrier, threads could read partially updated values → race condition. The pattern enforces a strict global read-then-write ordering on a consistent shared-memory snapshot.

**Multi-block case (`N > BLOCK_SIZE`).** Shared memory can't span blocks, so: divide the array into block-sized subarrays, run the single-block scan on each, then extract a **block-sums array** (length $\lceil N / \text{BLOCK\_SIZE} \rceil$) holding each block's total. You can't just add raw block sums to the next block — you must first **prefix-scan the block-sums array itself**, then `offset_add_kernel` adds `blockSums[blockIdx-1]` to every element of block `blockIdx`. The author shows the modified `prefix_kernel` (extra `block_sums` arg, written by the last thread in each block) and the `solve()` host code with `cudaMalloc` for the scratch arrays.

**The recursion.** The above assumes the block-sums array itself fits in one block (`blocks ≤ BLOCK_SIZE`) — often false. The fix is to apply the *same* divide-and-conquer recursively: scan blocks locally + extract sums, then recursively scan the sums; if those still overflow one block, recurse again. Each step shrinks the problem by a factor of `BLOCK_SIZE`, so recursion depth is $O(\log_{\text{BLOCK\_SIZE}} N)$ — tiny in practice.

## Notable details

- **Two conceptual phases** mirror a reduction tree: an *upward* phase (block-local scans → extract sums → shrink) and a *downward fix-up* phase (propagate recursively-computed block prefixes back, added in parallel). This is exactly the structure that avoids any cross-block synchronization.
- **Out-of-bounds handling**: threads with `tid >= N` initialize their shared cell to `0.0f` so they contribute nothing; writes are guarded by `if (tid < N)`.
- **Honest caveat on efficiency** (the most important line for a reader): the Hillis–Steele scan does $O(N \log N)$ *total* operations, not $O(N)$. Production scans use **Blelloch's work-efficient algorithm** ($O(N)$ work, up-sweep/down-sweep on a balanced tree).

## Why it matters [analyst's view]

Scan is one of the two canonical "parallel primitives" (with reduction) and the mental model here — *local compute in shared memory → extract coarse summary → recurse on the summary → broadcast the fix-up back* — is the reusable pattern, far more than the specific kernel. It's the same divide-and-conquer skeleton behind segmented scans, radix sort, stream compaction, and (relevant to the rest of this vault) the block-wise reductions inside softmax / layernorm / flash-attention kernels. Pairing this with [[blogs/bierling-coalesced-matmul]] from the same series gives the two halves of GPU-kernel intuition: *coalescing* (how threads should touch global memory) and *scan/reduction* (how threads should cooperate through shared memory). The author flagging Blelloch as the work-efficient successor is the right pointer to chase next.

## Connections

- Topic MOCs: [[topics/cuda-programming]], [[topics/parallel-algorithms]]
- Same series: [[blogs/bierling-coalesced-matmul]] (coalesced memory access, the companion kernel)
- Foundational concepts: [[blogs/aiwithmaha-cuda-concepts]] (threads/blocks/grid, shared memory, `__syncthreads` mental model)
- Author index: [[authors/lukas-bierling]]

## Selected quotes

> "What initially appeared to be a purely sequential problem can thus be transformed into a scalable parallel algorithm by exploiting the hierarchical execution model of CUDA and applying the same strategy recursively."

> "This implementation prioritizes clarity over optimal work efficiency. It uses a Hillis–Steele scan, which performs O(N log N) total operations. Production-grade scans typically use Blelloch's algorithm."
