from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from ragnar.models.chunk import Chunk
from ragnar.models.document import Document


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk the document into smaller pieces.

        Args:
            document (Document): The document to be chunked.

        Returns:
            list[Chunk]: A list of chunked documents.
        """
