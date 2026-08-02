from __future__ import annotations

import asyncio

from httpx import AsyncClient
from loguru import logger

from ragnar.embeddings.interfaces.base_embedding import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    def __init__(
        self, model: str = 'nomic-embed-text',
        max_concurrent: int = 3,
    ):
        self.model = model
        self.base_url = 'http://localhost:11434'
        self.client = AsyncClient(base_url=self.base_url)
        self.max_concurrent = asyncio.Semaphore(max_concurrent)

    async def embed(self, text: list[str]) -> list[list[float]]:
        """Generate an embedding for the given text using Ollama.

        Args:

            text (list[str]): The input text to be embedded.

        Returns:
            list[list[float]]: The generated embedding vectors.
        """
        async with self.max_concurrent:
            for attempt in range(3):  # Retry up to 3 times
                try:
                    response = await self.client.post(
                        '/api/embed',
                        json={'model': self.model, 'input': text},
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    return response.json()['embeddings']
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == 2:  # last attempt, raise the exception
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            raise RuntimeError('Embedding failed after all retries')
