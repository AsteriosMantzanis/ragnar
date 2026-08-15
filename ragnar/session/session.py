from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any

from ragnar.session.message import Message


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)

    def add_message(
        self,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        grounding: list[dict[str, Any]] | None = None,
    ):
        self.messages.append(
            Message(
                role=role,
                content=content,
                timestamp=datetime.now(timezone.utc),
                sources=sources,
                grounding=grounding,
            ),
        )

    def get_history(self, max_messages: int = 6) -> str:
        """Return formatted conversation history"""
        history_messages = self.messages[-max_messages:]
        return '\n'.join([
            f"{m.role.upper()}: {m.content}"
            for m in history_messages
        ])
