from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    element_type: str
    metadata: dict[str, Any]
