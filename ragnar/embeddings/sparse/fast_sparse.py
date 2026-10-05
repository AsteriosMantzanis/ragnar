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
        max_concurrent: int = 1,
        batch_size: int = int(os.getenv('SPARSE_EMBED_BATCH_SIZE', '8')),
        threads: int = int(os.getenv('SPARSE_EMBED_THREADS', '2')),
    ):
        if max_concurrent < 1:
            raise ValueError('max_concurrent must be >= 1')
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        if threads < 1:
            raise ValueError('threads must be >= 1')

        self.model = model
        self.max_concurrent = asyncio.Semaphore(max_concurrent)
        self.batch_size = batch_size
        self.embedder = SparseTextEmbedding(
            model_name=self.model,
            threads=threads,
        )

    @property
    def dimension(self) -> int:
        return 0

    async def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[SparseVector]:
        """Embed texts in sequential, bounded-memory batches."""
        if not texts:
            return []

        effective_batch_size = batch_size or self.batch_size
        if effective_batch_size < 1:
            raise ValueError('batch_size must be >= 1')

        results: list[SparseVector] = []

        for start in range(0, len(texts), effective_batch_size):
            batch = texts[start:start + effective_batch_size]
            batch_number = start // effective_batch_size + 1
            total_batches = (
                (len(texts) + effective_batch_size - 1)
                // effective_batch_size
            )
            logger.debug(
                f'Embedding sparse batch {batch_number}/{total_batches} '
                f'({len(batch)} texts)',
            )
            results.extend(await self._embed_batch(batch))

        return results

    async def _embed_batch(self, batch: list[str]) -> list[SparseVector]:
        """Generate sparse embeddings for exactly one bounded batch."""
        async with self.max_concurrent:
            for attempt in range(3):
                try:
                    embeddings = list(self.embedder.passage_embed(batch))
                    return [
                        SparseVector(
                            indices=e.indices.tolist(),
                            values=e.values.tolist(),
                        )
                        for e in embeddings
                    ]
                except Exception as exc:
                    logger.error(
                        'Sparse embedding attempt '
                        f'{attempt + 1} failed: {exc}',
                    )
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError('Embedding failed after all retries')

    async def embed_query(self, query: str) -> SparseVector:
        """Generate a sparse embedding for a query."""
        async with self.max_concurrent:
            embeddings = list(self.embedder.query_embed(query))

        return SparseVector(
            indices=embeddings[0].indices.tolist(),
            values=embeddings[0].values.tolist(),
        )
