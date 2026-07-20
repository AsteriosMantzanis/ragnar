from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from models.document import Document


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        pass
