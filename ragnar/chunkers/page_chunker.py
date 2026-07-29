from __future__ import annotations

from ragnar.chunkers.interfaces.base_chunker import BaseChunker
from ragnar.models.chunk import Chunk
from ragnar.models.document import Document
from ragnar.models.document_element import DocumentElement


class PageChunker(BaseChunker):
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunk the document into pages based on page numbers.
        Keep the page text and all following elements until the next page.

        Args:
            document (Document): The document to be chunked.

        Returns:
            list[Chunk]: A list of chunked documents.
        """
        pages = []
        buffer: list[DocumentElement] = []
        for element in document.elements:
            if element.page is not None:
                if buffer:
                    pages.extend(self.flush(buffer, document))
                    buffer = []
                buffer.append(element)
            else:
                buffer.append(element)

        # final flush of the buffer
        if buffer:
            pages.extend(self.flush(buffer, document))

        return pages

    def flush(
        self, buffer: list[DocumentElement],
        document: Document,
    ) -> list[Chunk]:
        """
        Flush the buffer to sections.

        Args:
            buffer (list[DocumentElement]): The buffer to be flushed.
            document (Document): The document to be chunked.
        Returns:
            list[Chunk]: A list of chunked documents.
        """
        text = f"[{document.metadata['filename']} — {buffer[0].text}]\n" + \
            '\n'.join([element.text for element in buffer[1:]])
        pages = [
            element.page for element in buffer
            if element.page is not None
        ]

        return [
            Chunk(
                id=document.id,
                text=text,
                pages=pages,
                metadata=document.metadata,
            ),
        ]
