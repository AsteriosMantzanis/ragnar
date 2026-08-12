from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseCrossEncoder(ABC):
    @abstractmethod
    async def score(
        self,
        query: str,
        results: list[dict],
    ) -> list[dict]:
        pass

    @abstractmethod
    async def score_hierarchical(
        self, query: str, results: dict[str, Any],
        child_entity: str,
    ) -> dict[str, Any]:
        pass
