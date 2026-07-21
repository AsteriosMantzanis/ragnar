from __future__ import annotations

from pathlib import Path

from loaders.interfaces.base_loader import BaseLoader
from models.document import Document


class DoclingLoader(BaseLoader):
    def load(self, path: Path) -> list[Document]:
        # TODO: finish this implementation
        docs = [Document(id='', text='', element_type='', metadata={})]
        return docs
