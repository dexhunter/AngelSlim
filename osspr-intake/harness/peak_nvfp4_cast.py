"""Peak RSS of ``nvfp4_cast_to_e2m1`` on one large linear weight.

The cast round-half-to-even check expands a 7-element tie mask against every
element (``weight.abs().unsqueeze(-1) == bounds``), which materializes a
temporary ``(*weight.shape, 7)`` bool tensor and a matching product. A local
rewrite can test the three odd E2M1 tie bounds without that broadcast.
"""

from __future__ import annotations

import torch

from angelslim.compressor.quant.modules.helper_layer import nvfp4_cast_to_e2m1

from _rss import peak_rss_delta_bytes


def main() -> None:
    torch.manual_seed(0)
    # One Qwen-scale projection: out_features x in_features, fp32, CPU.
    weight = torch.randn(4096, 8192)
    peak = peak_rss_delta_bytes(lambda: nvfp4_cast_to_e2m1(weight))
    print(f"peak_memory_bytes {peak}")


if __name__ == "__main__":
    main()
