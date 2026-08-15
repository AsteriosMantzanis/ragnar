from __future__ import annotations

from ragnar.session.interfaces.session_store import BaseSessionStore
from ragnar.session.session import Session


class InMemorySessionStore(BaseSessionStore):
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def save(self, session: Session) -> None:
        self.sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
