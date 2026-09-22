from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from loguru import logger

from ragnar.llm.interfaces.llm_adapter import BaseLLMAdapter

load_dotenv()


class OllamaLLMAdapter(BaseLLMAdapter):
    def __init__(
        self, model: str = os.getenv('LLM_MODEL', 'gemma2:2b'),
        base_url: str = os.getenv('OLLAMA_URL', 'http://localhost:11434'),
    ):
        self.model = model
        self.client = httpx.AsyncClient(base_url=base_url)

    async def generate(
        self,
        prompt: str,
        format: dict[str, Any] | str | None = None,
    ) -> str | dict[str, Any]:
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
        }
        if format is not None:
            payload['format'] = format

        for attempt in range(3):
            try:
                response = await self.client.post(
                    '/api/generate',
                    json=payload,
                    timeout=120.0,
                )
                response.raise_for_status()
                result = response.json()
                content = result.get('response', '')

                if format is not None:
                    try:
                        return json.loads(content)
                    except (TypeError, json.JSONDecodeError):
                        return {'response': content}

                return content
            except httpx.HTTPError as e:
                logger.exception(
                    f"Attempt {attempt + 1} failed with {type(e).__name__}",
                )
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError('Failed to generate response after 3 attempts')
