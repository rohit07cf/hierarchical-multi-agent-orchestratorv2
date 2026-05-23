"""System-resource and event-loop health sampling.

Two production failure modes this catches:

1. **Memory creep / leaks** — long-lived orchestrator processes that
   accumulate state (cached docs, unbounded buffers). RSS as a gauge
   makes the slope obvious.
2. **Event-loop starvation** — the silent killer of async systems. A
   synchronous CPU-bound tool call (or a blocking SDK) stalls the loop;
   every other coroutine waits. We measure the gap between *when a
   sleep(0) should have resumed* and when it actually did — that gap is
   head-of-line blocking, invisible to ordinary latency metrics.

Sampling runs as one background task started by ``init_observability``;
it is cancelled cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from src.observability.metrics import registry as M

logger = logging.getLogger(__name__)

try:
    import psutil

    _PROC = psutil.Process(os.getpid())
except Exception:  # noqa: BLE001 — psutil optional
    _PROC = None


async def sample_system_metrics(interval_seconds: float = 5.0) -> None:
    """Background loop sampling RSS, CPU, and event-loop lag.

    Loop lag = measured sleep duration − requested duration. On a healthy
    loop this is ~0; under starvation it climbs, pinpointing blocking code.
    """
    if _PROC is not None:
        _PROC.cpu_percent(None)  # prime the first (always-0) reading
    while True:
        start = time.perf_counter()
        await asyncio.sleep(interval_seconds)
        actual = time.perf_counter() - start
        M.EVENT_LOOP_LAG.observe(max(0.0, actual - interval_seconds))

        if _PROC is not None:
            try:
                M.PROCESS_MEMORY_BYTES.set(_PROC.memory_info().rss)
                M.PROCESS_CPU_PERCENT.set(_PROC.cpu_percent(None))
            except Exception:  # noqa: BLE001
                pass
