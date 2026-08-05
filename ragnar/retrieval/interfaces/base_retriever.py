from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Flat search — return chunk payloads."""

    @abstractmethod
    async def retrieve_hierarchical(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Hierarchical search — return sections + parent pages."""
