from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentElement:
    text: str
    label: str
    level: int
    page: int | None
    parent_ref: str | None
    element_id: str
