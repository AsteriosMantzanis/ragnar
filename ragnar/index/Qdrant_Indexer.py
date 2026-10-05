from __future__ import annotations

import hashlib
import json
import os
import uuid

from dotenv import load_dotenv
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import SparseVectorParams
from qdrant_client.http.models import VectorParams

from ragnar.embeddings.interfaces.base_embedding import BaseEmbedding
from ragnar.index.interfaces.base_indexer import BaseIndexer
from ragnar.models.chunk import Chunk

load_dotenv()


def point_id_for_chunk(collection_name: str, chunk: Chunk) -> int | str:
    try:
        chunk_uuid = uuid.UUID(chunk.id)
    except (AttributeError, ValueError):
        chunk_uuid = None

    if chunk_uuid is not None and chunk_uuid.version == 5:
        return int(hashlib.md5(chunk.id.encode()).hexdigest(), 16) % (10**8)

    identity = json.dumps(
        {
            'collection': collection_name,
            'text': chunk.text,
            'pages': chunk.pages,
            'metadata': chunk.metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    digest = hashlib.sha256(identity.encode('utf-8')).digest()
    return str(uuid.UUID(bytes=digest[:16], version=5))


class QdrantIndexer(BaseIndexer):
    def __init__(
        self,
        dense_embedder: BaseEmbedding,
        sparse_embedder: BaseEmbedding | None,
        collection_name: str,
        url: str = os.getenv('QDRANT_URL', 'http://localhost:6333'),
        batch_size: int = int(os.getenv('INDEX_BATCH_SIZE', '16')),
    ):
        if batch_size < 1:
            raise ValueError('INDEX_BATCH_SIZE must be >= 1')

        self.client = QdrantClient(url=url)
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.collection_name = collection_name
        self.batch_size = batch_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' exists.")
        except Exception as exc:
            logger.error(
                f"No collection called '{self.collection_name}': {exc}",
            )
            logger.warning(
                f"Creating collection '{self.collection_name}'",
            )
            collection_kwargs = {
                'collection_name': self.collection_name,
                'vectors_config': {
                    'text-dense': VectorParams(
                        size=self.dense_embedder.dimension,
                        distance=Distance.COSINE,
                    ),
                },
            }

            if self.sparse_embedder:
                collection_kwargs['sparse_vectors_config'] = {
                    'text-sparse': SparseVectorParams(),
                }

            self.client.create_collection(**collection_kwargs)

    async def index(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks in bounded batches.

        Peak memory is now roughly one index batch rather than the entire
        document/chunk list worth of vectors and PointStructs.
        """
        if not chunks:
            return

        total = len(chunks)
        logger.info(
            f'Indexing {total} chunks in batches of {self.batch_size}',
        )

        for start in range(0, total, self.batch_size):
            batch = chunks[start:start + self.batch_size]
            batch_number = start // self.batch_size + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

            texts = [chunk.text for chunk in batch]

            logger.debug(
                f'Index batch {batch_number}/{total_batches} '
                f'({len(batch)} chunks)',
            )

            dense_embeddings = await self.dense_embedder.embed(
                texts,
                batch_size=self.batch_size,
            )

            sparse_embeddings = None
            if self.sparse_embedder is not None:
                sparse_embeddings = await self.sparse_embedder.embed(
                    texts,
                    batch_size=self.batch_size,
                )

            if len(dense_embeddings) != len(batch):
                raise RuntimeError(
                    'Dense embedder returned an unexpected number of '
                    f'embeddings: expected {len(batch)}, '
                    f'got {len(dense_embeddings)}',
                )

            if (
                sparse_embeddings is not None
                and len(sparse_embeddings) != len(batch)
            ):
                raise RuntimeError(
                    'Sparse embedder returned an unexpected number of '
                    f'embeddings: expected {len(batch)}, '
                    f'got {len(sparse_embeddings)}',
                )

            points: list[PointStruct] = []
            for idx, chunk in enumerate(batch):
                dense_emb = dense_embeddings[idx]
                sparse_emb = (
                    sparse_embeddings[idx]
                    if sparse_embeddings is not None
                    else None
                )

                vector = {'text-dense': dense_emb}
                if sparse_emb is not None:
                    vector['text-sparse'] = sparse_emb

                points.append(
                    PointStruct(
                        id=point_id_for_chunk(self.collection_name, chunk),
                        vector=vector,
                        payload={
                            'chunk_id': chunk.id,
                            'text': chunk.text,
                            'pages': chunk.pages,
                            **chunk.metadata,
                        },
                    ),
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.debug(
                f'Upserted index batch {batch_number}/{total_batches}',
            )

            # Release batch-local references before the next batch.
            del texts
            del dense_embeddings
            del sparse_embeddings
            del points
            del batch

        logger.info(f'Indexed {total} chunks.')
