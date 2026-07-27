from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DocumentNode:
    element_type: str
    text: str
    metadata: dict[str, Any]
    children: list[DocumentNode]
