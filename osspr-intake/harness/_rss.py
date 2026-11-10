"""Peak-RSS delta while a callable runs.

Torch's CPU caching allocator returns large temporaries to the OS when the
call ends, so a sample taken after the function returns misses the spike.
VmRSS is sampled on a side thread for the duration of the call.
"""

from __future__ import annotations

import gc
import threading


def _vm_rss_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS missing from /proc/self/status")


def peak_rss_delta_bytes(fn) -> int:
    """Return max VmRSS during ``fn()`` minus VmRSS just before it."""
    gc.collect()
    baseline = _vm_rss_bytes()
    high = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            current = _vm_rss_bytes()
            if current > high[0]:
                high[0] = current

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    try:
        fn()
    finally:
        stop.set()
        watcher.join()
    return high[0] - baseline
