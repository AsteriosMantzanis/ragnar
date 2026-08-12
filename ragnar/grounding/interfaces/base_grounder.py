from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseGrounding(ABC):
    @abstractmethod
    async def ground(
        self, answer: str,
        sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def ground_hierarchical(
        self,
        answer: str,
        sources: dict[str, list[dict[str, Any]]],
        grounding_entity: str,
    ) -> list[dict[str, Any]]:
        pass
