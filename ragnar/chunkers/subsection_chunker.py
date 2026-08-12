from __future__ import annotations

import uuid

from ragnar.chunkers.interfaces.base_chunker import BaseChunker
from ragnar.models.chunk import Chunk
from ragnar.models.document import Document
from ragnar.models.document_element import DocumentElement


class SubsectionChunker(BaseChunker):
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunk document into subsections where each list_item is a subsection.
        Links subsections to parent section via section_id.

        Args:
            document (Document): The document to be chunked.

        Returns:
            list[Chunk]: A list of chunked subsections.
        """
        subsections = []
        buffer: list[DocumentElement] = []
        current_section_title: str | None = None
        current_section_id: str | None = None

        for element in document.elements:
            # New section starts
            if element.label.startswith('section'):
                if buffer:
                    subsections.extend(
                        self.flush(
                            buffer, document,
                            current_section_title, current_section_id,
                        ),
                    )
                    buffer = []

                # Compute section_id deterministically
                section_key = (
                    f"{document.metadata['filename']}_section_"
                    f"{element.text[:50]}"
                )
                current_section_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, section_key),
                )
                current_section_title = element.text
                buffer.append(element)
            # List item creates subsection boundary
            elif element.label == 'list_item':
                if buffer and buffer[0].label != 'section':
                    subsections.extend(
                        self.flush(
                            buffer,
                            document,
                            current_section_title,
                            current_section_id,
                        ),
                    )
                    buffer = []
                buffer.append(element)
            else:
                buffer.append(element)

        # Final flush
        if buffer:
            subsections.extend(
                self.flush(
                    buffer,
                    document,
                    current_section_title,
                    current_section_id,
                ),
            )

        return subsections

    def flush(
        self,
        buffer: list[DocumentElement],
        document: Document,
        section_title: str | None,
        section_id: str | None,
    ) -> list[Chunk]:
        """Flush buffer to subsection chunk."""
        text = '\n'.join([element.text for element in buffer])
        pages = [
            element.page for element in buffer if element.page is not None
        ]

        # Deterministic subsection ID
        subsection_key = (
            f"{document.metadata['filename']}_"
            f"{section_title}_{buffer[0].text[:30]}"
        )
        subsection_id = str(uuid.uuid5(uuid.NAMESPACE_URL, subsection_key))

        return [
            Chunk(
                id=subsection_id,
                text=text,
                pages=pages,
                metadata={
                    **document.metadata,
                    'section_title': section_title,
                    # make it a list so we consistently
                    # use set().update in retrieval
                    'parent_section_id': [section_id],
                    'subsection_title': (
                        buffer[0].text
                        if buffer[0].label == 'list_item'
                        else section_title
                    ),
                },
            ),
        ]
