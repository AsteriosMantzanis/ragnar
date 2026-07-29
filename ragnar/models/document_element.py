from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentElement:
    text: str
    label: str
    page: int | None
