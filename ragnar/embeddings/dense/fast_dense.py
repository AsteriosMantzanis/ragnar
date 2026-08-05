from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from fastembed import TextEmbedding
from loguru import logger

from ragnar.embeddings.interfaces.base_embedding import BaseEmbedding
load_dotenv()


class DenseFastEmbedding(BaseEmbedding):
    def __init__(
        self, model: str =
        os.getenv('dense_embed_model', 'nomic-ai/nomic-embed-text-v1.5-Q'),
        max_concurrent: int = 3,
    ):
        self.model = model
        self.max_concurrent = asyncio.Semaphore(max_concurrent)
        self.embedder = TextEmbedding(model_name=self.model)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the model."""
        return self.embedder.embedding_size()

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
                    embeddings = list(self.embedder.passage_embed(batch))
                    return embeddings
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == 2:  # last attempt, raise the exception
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            raise RuntimeError('Embedding failed after all retries')
