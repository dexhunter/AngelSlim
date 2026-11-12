"""Time ``vecattention_reference_prefill`` after the keep-mask build.

``build_vecattention_keep_mask`` is already batched. This prefill still
materializes a full ``(L, L)`` score from that mask, softmaxes, and multiplies
by V. ``scaled_dot_product_attention`` with the same boolean mask skips the
square score.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import statistics
import time

import torch

from angelslim.compressor.sparsity.algorithms.vecattention.reference import (
    vecattention_reference_prefill,
)


def _inputs():
    torch.manual_seed(0)
    # 2k prefill, 8 heads, head dim 128. threshold 0.5 is the MinP setting the
    # reference oracle uses to drop columns. The mask build is the fast path
    # landed in #395; this times the attention that follows it.
    query = torch.randn(1, 8, 2048, 128)
    key = torch.randn(1, 8, 2048, 128)
    value = torch.randn(1, 8, 2048, 128)
    return query, key, value


def _time_once(query, key, value) -> float:
    started = time.perf_counter()
    vecattention_reference_prefill(
        query,
        key,
        value,
        threshold=0.5,
        block_size_q=64,
        block_size_k=16,
    )
    return (time.perf_counter() - started) * 1000.0


def main() -> None:
    torch.set_num_threads(1)
    query, key, value = _inputs()
    _time_once(query, key, value)
    samples = [_time_once(query, key, value) for _ in range(7)]
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0
    print(f"median_ms {median:.6f}")
    print(f"noise_pct {noise:.6f}")


if __name__ == "__main__":
    main()
