from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Flat search — return chunk payloads."""

    @abstractmethod
    async def retrieve_hierarchical(
        self,
        query: str,
        parent_collection_name: str,
        child_collection_name: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Hierarchical search — return sections + parent pages."""
