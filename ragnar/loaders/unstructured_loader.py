from __future__ import annotations

from pathlib import Path

from loaders.interfaces.base_loader import BaseLoader
from models.document import Document


class UnstructuredLoader(BaseLoader):
    def load(self, path: Path) -> list[Document]:
        # TODO: finish this implementation
        documents = [Document(id='', text='', element_type='', metadata={})]
        return documents
