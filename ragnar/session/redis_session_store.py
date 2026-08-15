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
    def __init__(self, redis_url: str):
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.client: redis.Redis | None = None

    async def connect(self):
        self.client = await redis.from_url(self.redis_url)
        logger.info(f"Connected to Redis: {self.redis_url}")

    async def get(self, session_id: str) -> Session | None:
        assert self.client is not None, 'Redis not connected. '
        'Call connect() first.'

        data = await self.client.get(f"session:{session_id}")
        if not data:
            return None

        session_dict = json.loads(data)
        messages = [
            Message(
                role=m['role'],
                content=m['content'],
                timestamp=datetime.fromisoformat(m['timestamp']),
                sources=m.get('sources'),
                grounding=m.get('grounding'),
            )
            for m in session_dict['messages']
        ]

        session = Session(session_id=session_id)
        session.messages = messages
        return session

    async def save(self, session: Session) -> None:
        assert self.client is not None, 'Redis not connected. '
        'Call connect() first.'

        session_dict = {
            'session_id': session.session_id,
            'messages': [
                {
                    'role': m.role,
                    'content': m.content,
                    'timestamp': m.timestamp.isoformat(),
                    'sources': m.sources,
                    'grounding': m.grounding,
                }
                for m in session.messages
            ],
        }

        await self.client.set(
            f"session:{session.session_id}",
            json.dumps(session_dict),
            ex=86400 * 7,
        )

    async def delete(self, session_id: str) -> None:
        assert self.client is not None, 'Redis not connected. '
        'Call connect() first.'
        await self.client.delete(f"session:{session_id}")
