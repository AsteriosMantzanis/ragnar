from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from ragnar.session.session import Session


class BaseSessionStore(ABC):
    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        pass

    @abstractmethod
    async def save(self, session: Session) -> None:
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        pass
