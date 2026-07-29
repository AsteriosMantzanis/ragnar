from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    pages: list[int]
    metadata: dict[str, Any]
