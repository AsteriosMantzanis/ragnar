from __future__ import annotations

from uuid import uuid4

from ragnar.chunkers.interfaces.base_chunker import BaseChunker
from ragnar.models.chunk import Chunk
from ragnar.models.document import Document
from ragnar.models.document_element import DocumentElement


class FixedChunker(BaseChunker):
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunk the document into fixed-size chunks based
        on a specified number of elements.
        Keep the chunk text and all following elements until the next chunk.

        Args:
            document (Document): The document to be chunked.
        Returns:
            list[Chunk]: A list of chunked documents.
        """
        chunks = []
        buffer: list[DocumentElement] = []
        chunk_size = 1000  # Define the fixed size for each chunk in characters
        overlap = 100  # Define the overlap size for each chunk in characters
        current_len = 0
        overlap_text = ''

        for element in document.elements:
            buffer.append(element)
            current_len += len(element.text)
            if current_len >= chunk_size:
                chunk = self.flush(buffer, document, overlap_text)[0]
                overlap_text = chunk.text[-overlap:] if len(
                    chunk.text,
                ) > overlap else ''
                chunks.append(chunk)
                buffer = []
                current_len = 0

        # final flush of the buffer
        if buffer:
            chunks.extend(self.flush(buffer, document, overlap_text))

        return chunks

    def flush(
        self, buffer: list[DocumentElement],
        document: Document,
        overlap_text: str,
    ) -> list[Chunk]:
        """
        Flush the buffer to chunks.

        Args:
            buffer (list[DocumentElement]): The buffer to be flushed.
            document (Document): The document to be chunked.
            overlap_text (str): The text to be included as overlap
                                in the next chunk.
        Returns:
            list[Chunk]: A list of chunked documents.
        """
        text = f"[{document.metadata['filename']}]\n" + overlap_text + '\n' + \
            '\n'.join([element.text for element in buffer])
        pages = [
            element.page for element in buffer if
            element.page is not None
        ]

        return [
            Chunk(
                id=uuid4().hex,
                text=text,
                pages=pages,
                metadata={
                    **document.metadata,  # add more metadata to the chunk
                    'chunk_type': 'fixed+overlap',
                },
            ),
        ]
