from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseLLMAdapter(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        format: dict[str, Any] | str | None = None,
    ) -> str | dict[str, Any]:
        pass
