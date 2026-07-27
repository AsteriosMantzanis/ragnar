from __future__ import annotations

import hashlib
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import TableItem
from loguru import logger

from ragnar.loaders.interfaces.base_loader import BaseLoader
from ragnar.models.document import Document
from ragnar.models.document_element import DocumentElement


class DoclingLoader(BaseLoader):
    def __init__(self):
        self.allowed_formats = ['pdf', 'docx']
        self.converter = DocumentConverter(
            allowed_formats=self.allowed_formats,
        )

    def load(self, path: Path) -> list[Document]:
        if path.is_file():
            return self._convert([path])

        files = [
            f for f in path.rglob(
                '*',
            ) if f.suffix.lstrip('.').lower() in self.allowed_formats
        ]
        return self._convert(files)

    def _convert(self, files: list[Path]) -> list[Document]:
        results = self.converter.convert_all(
            [str(i) for i in files], raises_on_error=False,
        )
        documents = []
        for result in results:
            documents.extend(self._to_ragnar_documents(result))

        return documents

    def _to_ragnar_documents(self, result) -> list[Document]:
        doc = result.document
        path = Path(result.input.file)

        elements = []
        for item, level in doc.iterate_items():
            if isinstance(item, TableItem):
                text = item.export_to_markdown(doc)
            else:
                text = getattr(item, 'text', None)

            # skip empty items
            if not text or not text.strip():
                continue

            element = DocumentElement(
                text=text,
                label=item.label.value,
                level=level,
                page=getattr(
                    item.prov[0], 'page_no',
                    None,
                ) if item.prov else None,
                parent_ref=item.parent,
                element_id=item.self_ref,
            )
            elements.append(element)
            logger.info(
                f"Processed element: {element.label} Text: {element.text}",
            )

        doc_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]

        return [
            Document(
                id=doc_id,
                source=str(path),
                elements=elements,
                metadata={
                    'filename': path.name,
                    'format': path.suffix.lstrip('.'),
                    'loader': 'DoclingLoader',
                    'element_count': len(elements),
                },
            ),
        ]
