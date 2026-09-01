from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone

from loguru import logger


@dataclass
class StepMetrics:
    name: str
    duration_s: float
    timestamp: str


@dataclass
class QueryMetrics:
    session_id: str
    query: str
    strategy: str
    total_duration_s: float
    steps: list[dict]
    answer_length: int
    num_sources: int
    grounding_percentage: float
    cache_hit: bool = False
    cache_score: float | None = None
    timestamp: str | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(
                timezone.utc,
            ).isoformat()


class MetricsCollector:
    """Collect and store query metrics"""

    def __init__(self):
        self.metrics: list[QueryMetrics] = []

    def record_query(self, metrics: QueryMetrics) -> None:
        """Record a query's metrics"""
        self.metrics.append(metrics)

        logger.info(
            f"Query metrics recorded | "
            f"Total: {metrics.total_duration_s:.0f}ms | "
            f"Sources: {metrics.num_sources} | "
            f"Grounding: {metrics.grounding_percentage:.0f}%",
        )
