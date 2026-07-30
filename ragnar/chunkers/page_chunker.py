from __future__ import annotations

import hashlib

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
        current_page: int | None = None

        for element in document.elements:
            if element.page != current_page and buffer:
                pages.extend(self.flush(buffer, document))
                buffer = []
            current_page = element.page
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
        Flush the buffer to pages.

        Args:
            buffer (list[DocumentElement]): The buffer to be flushed.
            document (Document): The document to be chunked.
        Returns:
            list[Chunk]: A list of chunked documents.
        """
        text = f"[{document.metadata['filename']} — {buffer[0].text}]\n" + \
            '\n'.join([element.text for element in buffer[1:]])
        page = buffer[0].page if buffer[0].page is not None else 0

        # create page ids from document name and page number
        page_id = hashlib.sha256(
            f"{document.metadata['filename']}_{page}".encode(),
        ).hexdigest()

        return [
            Chunk(
                id=page_id,
                text=text,
                pages=page,
                metadata={
                    **document.metadata,  # add more metadata to the chunk
                    'page_title': buffer[0].text,
                },
            ),
        ]
