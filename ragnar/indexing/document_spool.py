from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from ragnar.loaders.interfaces.base_loader import BaseLoader
from ragnar.models.document import Document
from ragnar.models.document_element import DocumentElement


def write_document(document: Document, output: TextIO) -> None:
    record = {
        'id': document.id,
        'elements': [
            {
                'text': element.text,
                'label': element.label,
                'page': element.page,
            }
            for element in document.elements
        ],
        'metadata': document.metadata,
    }
    output.write(json.dumps(record, ensure_ascii=False))
    output.write('\n')


class DocumentSpoolLoader(BaseLoader):
    def __init__(self, checkpoint_path: Path | None = None):
        self.checkpoint_path = checkpoint_path
        self._progress = self._read_progress()

    def load(self, path: Path) -> list[Document]:
        return list(self.iter_load(path))

    def iter_load(self, path: Path) -> Iterator[Document]:
        for _, document in self.iter_load_indexed(path):
            yield document

    def iter_load_indexed(
        self,
        path: Path,
    ) -> Iterator[tuple[int, Document]]:
        completed_records = self._progress.get('completed_records', 0)
        record_number = 0

        with Path(path).open(encoding='utf-8') as spool:
            for line_number, line in enumerate(spool, start=1):
                if not line.strip():
                    continue

                record_number += 1
                if record_number <= completed_records:
                    continue

                try:
                    record = json.loads(line)
                    yield record_number, Document(
                        id=record['id'],
                        elements=[
                            DocumentElement(
                                text=element['text'],
                                label=element['label'],
                                page=element['page'],
                            )
                            for element in record['elements']
                        ],
                        metadata=record['metadata'],
                    )
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f'Invalid document spool record at line {line_number}',
                    ) from exc

    def resume_progress(self) -> dict[str, int]:
        return dict(self._progress)

    def mark_completed(
        self,
        document_number: int,
        progress: dict[str, int],
    ) -> None:
        self._progress = {
            **progress,
            'completed_records': document_number,
        }
        if self.checkpoint_path is None:
            return

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = self.checkpoint_path.with_suffix('.json.partial')
        try:
            with partial_path.open('w', encoding='utf-8') as output:
                json.dump(self._progress, output, sort_keys=True)
                output.flush()
                os.fsync(output.fileno())
            os.replace(partial_path, self.checkpoint_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    def _read_progress(self) -> dict[str, int]:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return {'completed_records': 0}

        try:
            progress = json.loads(
                self.checkpoint_path.read_text(encoding='utf-8'),
            )
            completed_records = progress['completed_records']
            if not isinstance(completed_records, int) or completed_records < 0:
                raise ValueError(
                    'completed_records must be a non-negative int',
                )
            return progress
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f'Invalid indexing checkpoint: {self.checkpoint_path}',
            ) from exc
