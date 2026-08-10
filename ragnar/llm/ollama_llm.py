from __future__ import annotations

import asyncio
import os

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

    async def generate(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                response = await self.client.post(
                    '/api/generate',
                    json={
                        'model': self.model,
                        'prompt': prompt,
                        'stream': False,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()['response']
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError('Failed to generate response after 3 attempts')
