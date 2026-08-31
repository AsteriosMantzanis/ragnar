from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseGenerator(ABC):

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: list[dict[str, Any]],
        prompt_template: str = 'simple_qa',
    ) -> str:
        pass

    @abstractmethod
    async def generate_hierarchical(
        self,
        query: str,
        context: dict[str, list[dict[str, Any]]],
        generation_entity: str,
        prompt_template: str = 'simple_qa',
    ) -> str:
        pass
