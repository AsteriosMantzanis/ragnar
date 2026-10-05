from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path

from ragnar.models.document import Document


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        pass

    def iter_load(self, path: Path) -> Iterator[Document]:
        """Stream documents without changing the existing load() contract.

        Concrete loaders that can process one file at a time should override
        this method. The default implementation preserves backwards
        compatibility for loaders that only implement load().
        """
        yield from self.load(path)

    def iter_load_indexed(
        self,
        path: Path,
    ) -> Iterator[tuple[int, Document]]:
        yield from enumerate(self.iter_load(path), 1)

    def resume_progress(self) -> dict[str, int]:
        return {}

    def mark_completed(
        self,
        document_number: int,
        progress: dict[str, int],
    ) -> None:
        pass
