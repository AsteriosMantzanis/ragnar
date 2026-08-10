from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseQueryExpander(ABC):
    @abstractmethod
    async def expand(self, query: str) -> list[str]:
        pass
