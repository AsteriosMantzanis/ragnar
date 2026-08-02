from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from ragnar.models.chunk import Chunk


class BaseIndexer(ABC):
    @abstractmethod
    async def index(self, chunks: list[Chunk]) -> None:
        pass
