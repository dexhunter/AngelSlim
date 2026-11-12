"""Time ``_quant_fp4_element`` on one linear weight.

FOCUS NVFP4 and MXFP4 fake-quant call this on every weight. The even-rounding
branch builds ``torch.zeros_like`` and a ``% 2`` tie mask the size of the
weight. ``torch.round`` (half to even) plus the same straight-through wrapper
matches the forward and the gradient without that mask.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import statistics
import time

import torch

from angelslim.compressor.qat.modules.quantizer import _quant_fp4_element


def _weight() -> torch.Tensor:
    torch.manual_seed(0)
    # One Qwen-scale projection, fp32, the tensor QAT fake-quant snaps to E2M1.
    return torch.randn(4096, 8192)


def _time_once(weight: torch.Tensor) -> float:
    started = time.perf_counter()
    _quant_fp4_element(weight, ebits=2, mbits=3, max_norm=6.0, mantissa_rounding="even")
    return (time.perf_counter() - started) * 1000.0


def main() -> None:
    torch.set_num_threads(1)
    weight = _weight()
    _time_once(weight)
    samples = [_time_once(weight) for _ in range(5)]
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0
    print(f"median_ms {median:.6f}")
    print(f"noise_pct {noise:.6f}")


if __name__ == "__main__":
    main()
