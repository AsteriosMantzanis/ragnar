from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client.http.models import SparseVector

from ragnar.embeddings.interfaces.base_embedding import BaseEmbedding
load_dotenv()


class SparseFastEmbedding(BaseEmbedding):
    def __init__(
        self,
        model: str = os.getenv(
            'sparse_embed_model',
            'prithivida/Splade_PP_en_v1',
        ),
        max_concurrent: int = 3,
    ):
        self.model = model
        self.max_concurrent = asyncio.Semaphore(max_concurrent)
        self.embedder = SparseTextEmbedding(model_name=self.model)

    @property
    def dimension(self) -> int:
        return 0

    async def embed(
        self, texts: list[str],
        batch_size: int = 20,
    ) -> list[SparseVector]:
        """Split tasks in batch size and generate embeddings.

        Args:
            texts (list[str]): The input texts to be embedded.
            batch_size (int): The size of each batch for embedding.

        Returns:
            list[SparseVector]: The generated embedding vectors.
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

    async def _embed_batch(self, batch: list[str]) -> list[SparseVector]:
        """Generate an embedding for the given batch.

        Args:

            text (list[str]): The input text to be embedded.

        Returns:
            list[SparseVector]: The generated embedding vectors.
        """
        async with self.max_concurrent:
            for attempt in range(3):  # Retry up to 3 times
                try:
                    embeddings = list(self.embedder.passage_embed(batch))
                    return [
                        SparseVector(
                            indices=e.indices.tolist(),
                            values=e.values.tolist(),
                        ) for e in embeddings
                    ]
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == 2:  # last attempt, raise the exception
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            raise RuntimeError('Embedding failed after all retries')
