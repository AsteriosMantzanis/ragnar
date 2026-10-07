from __future__ import annotations

import json
import os
from datetime import datetime

import redis.asyncio as redis
from dotenv import load_dotenv
from loguru import logger

from ragnar.session.interfaces.session_store import BaseSessionStore
from ragnar.session.message import Message
from ragnar.session.session import Session
load_dotenv()


class RedisSessionStore(BaseSessionStore):
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv(
            'REDIS_URL', 'redis://localhost:6379',
        )
        self.client: redis.Redis | None = None

    async def connect(self):
        if self.client is None:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            await self.client.ping()
        logger.info(f"Connected to Redis: {self.redis_url}")

    async def disconnect(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _ensure_connected(self):
        if self.client is None:
            await self.connect()

    async def get(self, session_id: str) -> Session | None:
        await self._ensure_connected()

        client = self.client
        if client is None:
            raise RuntimeError('Redis client is not connected')

        data = await client.get(f"session:{session_id}")
        if not data:
            return None

        session_dict = json.loads(data)
        messages = [
            Message(
                role=m['role'],
                content=m['content'],
                timestamp=datetime.fromisoformat(m['timestamp']),
                sources=m.get('sources'),
            )
            for m in session_dict['messages']
        ]

        session = Session(session_id=session_id)
        session.messages = messages
        return session

    async def save(self, session: Session) -> None:
        await self._ensure_connected()

        client = self.client
        if client is None:
            raise RuntimeError('Redis client is not connected')

        session_dict = {
            'session_id': session.session_id,
            'messages': [
                {
                    'role': m.role,
                    'content': m.content,
                    'timestamp': m.timestamp.isoformat(),
                    'sources': m.sources,
                }
                for m in session.messages
            ],
        }

        await client.set(
            f"session:{session.session_id}",
            json.dumps(session_dict),
            ex=86400 * 7,
        )

    async def delete(self, session_id: str) -> None:
        await self._ensure_connected()

        client = self.client
        if client is None:
            raise RuntimeError('Redis client is not connected')

        await client.delete(f"session:{session_id}")
