from __future__ import annotations

import json
import os
import uuid

import redis.asyncio as redis
from loguru import logger


QUEUE_NAME = 'ragnar:index:queue'
JOB_PREFIX = 'ragnar:index:job:'
JOB_TTL = 86400  # 24 hours


class IndexJobQueue:
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv(
            'REDIS_URL',
            'redis://localhost:6379',
        )
        self.client: redis.Redis | None = None

    async def connect(self) -> None:
        if self.client is None:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=None,
                socket_connect_timeout=5,
            )
            await self.client.ping()
            logger.info(f"Index job queue connected: {self.redis_url}")

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def enqueue(self, path: str, strategy: str) -> str:
        await self.connect()

        if self.client is None:
            raise RuntimeError('Redis queue is not connected')

        job_id = uuid.uuid4().hex

        job = {
            'job_id': job_id,
            'path': path,
            'strategy': strategy,
            'status': 'queued',
        }

        key = f"{JOB_PREFIX}{job_id}"

        await self.client.hset(
            key,
            mapping={
                'job_id': job_id,
                'path': path,
                'strategy': strategy,
                'status': 'queued',
                'result': '',
                'error': '',
            },
        )
        await self.client.expire(key, JOB_TTL)

        await self.client.rpush(
            QUEUE_NAME,
            json.dumps(job),
        )

        logger.info(
            f"Index job queued | job_id={job_id} | "
            f"path={path} | strategy={strategy}",
        )

        return job_id

    async def get(self, job_id: str) -> dict | None:
        await self.connect()

        if self.client is None:
            raise RuntimeError('Redis queue is not connected')

        data = await self.client.hgetall(
            f"{JOB_PREFIX}{job_id}",
        )

        if not data:
            return None

        result = data.get('result', '')

        if result:
            try:
                data['result'] = json.loads(result)
            except json.JSONDecodeError:
                pass

        return data

    async def retry(self, job_id: str) -> None:
        await self.connect()

        if self.client is None:
            raise RuntimeError('Redis queue is not connected')

        job_key = f"{JOB_PREFIX}{job_id}"
        data = await self.client.hgetall(job_key)
        if not data:
            raise KeyError(f"Index job not found: {job_id}")

        job = {
            'job_id': job_id,
            'path': data['path'],
            'strategy': data['strategy'],
            'status': 'queued',
        }
        script = """
        if redis.call('EXISTS', KEYS[1]) == 0 then
            return -1
        end
        if redis.call('HGET', KEYS[1], 'status') ~= 'failed' then
            return 0
        end
        redis.call(
            'HSET', KEYS[1], 'status', 'queued', 'result', '', 'error', ''
        )
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        redis.call('RPUSH', KEYS[2], ARGV[1])
        return 1
        """
        result = await self.client.eval(
            script,
            2,
            job_key,
            QUEUE_NAME,
            json.dumps(job),
            JOB_TTL,
        )
        if result == -1:
            raise KeyError(f"Index job not found: {job_id}")
        if result == 0:
            raise ValueError(f"Index job is not failed: {job_id}")

    async def update(
        self,
        job_id: str,
        *,
        status: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        await self.connect()

        if self.client is None:
            raise RuntimeError('Redis queue is not connected')

        values = {'status': status}

        if result is not None:
            values['result'] = json.dumps(result)

        if error is not None:
            values['error'] = error

        await self.client.hset(
            f"{JOB_PREFIX}{job_id}",
            mapping=values,
        )
