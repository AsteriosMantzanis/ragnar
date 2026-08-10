from __future__ import annotations

import uuid

from ragnar.chunkers.interfaces.base_chunker import BaseChunker
from ragnar.models.chunk import Chunk
from ragnar.models.document import Document
from ragnar.models.document_element import DocumentElement


class SectionChunker(BaseChunker):
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunk the document into sections based on headings.
        Keep the section text and following elements until
        the next section.

        Args:
            document (Document): The document to be chunked.

        Returns:
            list[Chunk]: A list of chunked documents.
        """
        sections: list[Chunk] = []
        buffer: list[DocumentElement] = []
        for element in document.elements:
            if element.label.startswith('section'):
                if buffer:
                    sections.extend(self.flush(buffer, document))
                    buffer = []
                buffer.append(element)
            else:
                buffer.append(element)

        # final flush of the buffer
        if buffer:
            sections.extend(self.flush(buffer, document))

        return sections

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
            element.page for element in buffer if
            element.page is not None
        ]

        # create page ids from document name and page number
        page_keys = [
            f"{document.metadata['filename']}_{page}"
            for page in list(set(pages))
        ]
        page_ids = [
            str(
                uuid.
                uuid5(uuid.NAMESPACE_URL, page_key),
            )
            for page_key in page_keys
        ]

        # section ids
        section_key = (
            f"{document.metadata['filename']}_section_"
            f"{buffer[0].text[:50]}"
        )
        section_id = str(uuid.uuid5(uuid.NAMESPACE_URL, section_key))

        return [
            Chunk(
                id=section_id,
                text=text,
                pages=pages,
                metadata={
                    **document.metadata,  # add more metadata to the chunk
                    'parent_page_ids': page_ids,
                    'section_title': buffer[0].text,
                },
            ),
        ]
