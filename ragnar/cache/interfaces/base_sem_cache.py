from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseSemanticCache(ABC):

    @abstractmethod
    async def get(
        self,
        query: str,
        strategy: str,
    ) -> dict[str, Any] | None:
        """Return a cached response if a semantically similar query exists."""

    @abstractmethod
    async def set(
        self,
        query: str,
        strategy: str,
        response: dict[str, Any],
    ) -> None:
        """Store a query response in the semantic cache."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear the semantic cache."""
