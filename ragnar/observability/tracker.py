from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime

from ragnar.observability.metrics import StepMetrics


class QueryTracker:
    """Track query execution metrics"""

    def __init__(self):
        self.steps: list[dict] = []
        self.start_time: float | None = None

    def start(self) -> None:
        """Start tracking"""
        self.start_time = time.perf_counter()
        self.steps = []

    @asynccontextmanager
    async def step(self, name: str):
        """Track a pipeline step"""
        step_start = time.perf_counter()
        try:
            yield
        finally:
            duration_s = (time.perf_counter() - step_start)

            step_metric = StepMetrics(
                name=name,
                duration_s=duration_s,
                timestamp=datetime.now().isoformat(),
            )

            self.steps.append(asdict(step_metric))

    def end(self) -> float:
        """End tracking, return total duration"""
        if self.start_time is None:
            raise RuntimeError('Tracker has not been started.')

        return (time.perf_counter() - self.start_time)
