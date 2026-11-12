"""Time ``_masked_softmax_attention`` on one causal prefill.

``a_shape_attention``, ``tri_shape_attention`` and ``vertical_and_slash_attention``
all finish through this helper. It builds a dense ``(L, L)`` score, fills the
masked keys with ``-inf``, and softmaxes. ``scaled_dot_product_attention`` with
the same boolean keep mask matches the output and skips that materialized score.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import statistics
import time

import torch

from angelslim.compressor.sparsity.algorithms.minference.reference import (
    _masked_softmax_attention,
)


def _inputs():
    torch.manual_seed(0)
    # One CPU prefill the reference patterns actually run: 8 heads, 2k tokens,
    # head dim 128. The keep mask is the causal triangle those patterns pass
    # when the budget covers the sequence.
    query = torch.randn(1, 8, 2048, 128)
    key = torch.randn(1, 8, 2048, 128)
    value = torch.randn(1, 8, 2048, 128)
    length = query.shape[-2]
    keep = torch.ones(length, length, dtype=torch.bool).tril(0)
    return query, key, value, keep.view(1, 1, length, length)


def _time_once(query, key, value, keep) -> float:
    started = time.perf_counter()
    _masked_softmax_attention(query, key, value, keep)
    return (time.perf_counter() - started) * 1000.0


def main() -> None:
    torch.set_num_threads(1)
    query, key, value, keep = _inputs()
    _time_once(query, key, value, keep)
    samples = [_time_once(query, key, value, keep) for _ in range(7)]
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0
    print(f"median_ms {median:.6f}")
    print(f"noise_pct {noise:.6f}")


if __name__ == "__main__":
    main()
