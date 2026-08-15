from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime
    sources: list[dict] | None = None
    grounding: list[dict] | None = None
