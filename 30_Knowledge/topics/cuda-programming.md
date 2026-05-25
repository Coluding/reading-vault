---
type: moc
topic: cuda-programming
last_updated: 2026-05-25
---

# CUDA Programming

The GPU execution model and how to write kernels against it: the threads → blocks → grid hierarchy, warps (32 threads in lockstep), the global/shared memory split, and the synchronization primitives (`__syncthreads`) that make cooperative algorithms correct. This MOC collects the foundational concepts plus worked algorithm examples from the "CUDA Algorithm Series." For the *performance/efficiency* angle (coalescing, occupancy, mixed precision, inference memory) see the overlapping [[topics/gpu-optimization]].

## Foundational
- [[blogs/aiwithmaha-cuda-concepts]] — beginner glossary of 20 CUDA concepts (kernel/grid/block/thread/warp hierarchy, global vs shared memory, occupancy, streams, tensor cores, cuDNN). The shared-vocabulary entry point.

## Algorithms (CUDA Algorithm Series)
- [[blogs/bierling-prefix-scan]] — parallel prefix sum (scan): single-block Hillis–Steele scan in shared memory ($O(\log N)$ depth) extended to arbitrary sizes via divide-and-conquer recursion over block sums.
- [[blogs/bierling-coalesced-matmul]] — naive vs coalesced matmul; a single thread-index remap turns strided global loads into coalesced ones for ~10.5× on an A100.

## Related topics
- [[topics/gpu-optimization]]
- [[topics/parallel-algorithms]]
