from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.document_element import DocumentElement


@dataclass
class Document:
    id: str
    source: str
    elements: list[DocumentElement]
    metadata: dict[str, Any]
