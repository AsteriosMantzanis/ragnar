from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseLLMAdapter(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
    ) -> str:
        pass
