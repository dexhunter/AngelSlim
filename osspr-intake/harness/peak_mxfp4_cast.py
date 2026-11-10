"""Peak RSS of ``mxfp4_cast_to_e2m1`` on one large linear weight.

Same E2M1 tie rule as the NVFP4 cast, implemented separately: the round-up
check broadcasts all 7 bounds against every magnitude. A local equality
against the three odd tie levels keeps the codes identical and drops the
``(*shape, 7)`` temporary.
"""

from __future__ import annotations

import torch

from angelslim.compressor.quant.modules.mxfp4.mxfp4 import mxfp4_cast_to_e2m1

from _rss import peak_rss_delta_bytes


def main() -> None:
    torch.manual_seed(0)
    weight = torch.randn(4096, 8192)
    peak = peak_rss_delta_bytes(lambda: mxfp4_cast_to_e2m1(weight))
    print(f"peak_memory_bytes {peak}")


if __name__ == "__main__":
    main()
