"""Time ``build_vecattention_keep_mask`` on a 4k multi-head prefill.

Each query block mean-pools its rows and matmuls that vector against every
key inside a Python loop. One batched matmul over the block axis matches the
mask exactly and removes the loop.
"""

from __future__ import annotations

import statistics
import time

import torch

from angelslim.compressor.sparsity.algorithms.vecattention.reference import (
    build_vecattention_keep_mask,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    # Qwen-like head size, 4k prefill, several heads. The mask is the
    # pseudo-sparse fallback used when the vLLM kernel is absent (head_dim
    # 256 on Qwen3.5 is the production case; 128 is the kernel-sized sibling
    # and exercises the same loop).
    query = torch.randn(1, 8, 4096, 128)
    key = torch.randn(1, 8, 4096, 128)
    return query, key


def _time_once(query: torch.Tensor, key: torch.Tensor) -> float:
    started = time.perf_counter()
    build_vecattention_keep_mask(
        query,
        key,
        threshold=0.5,
        block_size_q=64,
        block_size_k=16,
    )
    return (time.perf_counter() - started) * 1000.0


def main() -> None:
    torch.set_num_threads(4)
    query, key = _inputs()
    _time_once(query, key)  # warmup
    samples = [_time_once(query, key) for _ in range(7)]
    print(f"median_ms {statistics.median(samples):.6f}")


if __name__ == "__main__":
    main()
