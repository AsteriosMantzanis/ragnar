from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        pass
