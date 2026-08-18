from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    async def generate(
        self,
        query: str,
        context: list[dict],
        prompt_template: str,
    ) -> str:
        pass

    @abstractmethod
    async def generate_hierarchical(
        self,
        query: str,
        context: list[dict],
        prompt_template: str,
        generation_entity: str,
    ) -> str:
        pass
