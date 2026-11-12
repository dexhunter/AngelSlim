"""Peak RSS of ``nvfp4_cast_to_e2m1`` on one large linear weight.

The round-half-to-even check expands a 7-element tie mask against every
element (``weight.abs().unsqueeze(-1) == bounds``), which materializes a
temporary ``(*weight.shape, 7)`` bool tensor and a matching product. Comparing
the three odd E2M1 tie levels (0.75, 1.75, 3.5) matches the uint4 codes
without that broadcast.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(__file__))

import torch

from angelslim.compressor.quant.modules.helper_layer import nvfp4_cast_to_e2m1

from _rss import peak_rss_delta_bytes


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    # One Qwen-scale projection: out_features x in_features, fp32, CPU.
    weight = torch.randn(4096, 8192)
    samples = [peak_rss_delta_bytes(lambda: nvfp4_cast_to_e2m1(weight)) for _ in range(5)]
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0
    print(f"baseline_value {int(median)}")
    print(f"noise_pct {noise:.6f}")


if __name__ == "__main__":
    main()
