from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseEmbedding(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension, or None for sparse embedders."""

    @abstractmethod
    async def embed(self, text: list[str]) -> list[list[float]]:
        """Generate an embedding for the given text.

        Args:
            text (list[str]): The input text to be embedded.

        Returns:
            list[list[float]]: The generated embedding vectors.
        """
