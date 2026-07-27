from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]
