"""Time ``flashprefill_reference_prefill`` on a 2k block-sparse prefill.

The keep mask from ``build_block_keep_mask`` is cheap. The reference then
materializes a full ``(L, L)`` score, softmaxes it, and multiplies by V.
``scaled_dot_product_attention`` with that same boolean mask does the attention
without the square temporary.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import statistics
import time

import torch

from angelslim.compressor.sparsity.algorithms.flashprefill.reference import (
    flashprefill_reference_prefill,
)


def _inputs():
    torch.manual_seed(0)
    # Qwen-like 2k prefill, bf16, the dtype the reference serves for head_dim
    # 256 and the dtype this 128-wide sibling uses on CPU. alpha 0.5 is a
    # real sparsity setting (alpha 0 keeps every causal block).
    query = torch.randn(1, 8, 2048, 128, dtype=torch.bfloat16)
    key = torch.randn(1, 8, 2048, 128, dtype=torch.bfloat16)
    value = torch.randn(1, 8, 2048, 128, dtype=torch.bfloat16)
    return query, key, value


def _time_once(query, key, value) -> float:
    started = time.perf_counter()
    flashprefill_reference_prefill(
        query,
        key,
        value,
        alpha=0.5,
        block_size=128,
        sink=128,
        window=256,
        last_n_block_full=2,
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
