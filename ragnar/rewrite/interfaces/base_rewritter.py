from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseQueryRewriter(ABC):
    @abstractmethod
    async def rewrite(self, query: str) -> str:
        pass
