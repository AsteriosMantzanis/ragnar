from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from httpx import AsyncClient
from loguru import logger

from ragnar.embeddings.interfaces.base_embedding import BaseEmbedding
load_dotenv()


class OllamaEmbedding(BaseEmbedding):
    def __init__(
        self, model: str = os.getenv('EMBED_MODEL', 'nomic-embed-text:latest'),
        max_concurrent: int = 3,
    ):
        self.model = model
        self.base_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.client = AsyncClient(base_url=self.base_url)
        self.max_concurrent = asyncio.Semaphore(max_concurrent)

    async def embed(
        self, texts: list[str],
        batch_size: int = 20,
    ) -> list[list[float]]:
        """Split tasks in batch size and generate embeddings.

        Args:
            texts (list[str]): The input texts to be embedded.
            batch_size (int): The size of each batch for embedding.

        Returns:
            list[list[float]]: The generated embedding vectors.
        """
        batches = [
            texts[i:i+batch_size]
            for i in range(0, len(texts), batch_size)
        ]

        tasks = [self._embed_batch(batch) for batch in batches]

        results = await asyncio.gather(*tasks)

        return [
            emb for batch_embeddings in results
            for emb in batch_embeddings
        ]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Generate an embedding for the given batch.

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
                        json={'model': self.model, 'input': batch},
                        timeout=10.0,
                    )
                    response.raise_for_status()
                    return response.json()['embeddings']
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == 2:  # last attempt, raise the exception
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            raise RuntimeError('Embedding failed after all retries')
