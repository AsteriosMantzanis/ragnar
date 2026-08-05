from __future__ import annotations

import hashlib
import os

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


class QdrantIndexer(BaseIndexer):
    def __init__(
        self,
        dense_embedder: BaseEmbedding,
        sparse_embedder: BaseEmbedding,
        collection_name: str,
        url: str = os.getenv('QDRANT_URL', 'http://localhost:6333'),
    ):
        self.client = QdrantClient(url=url)
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' exists.")
        except Exception as e:
            logger.error(
                f"No collection called '{self.collection_name}': {e}",
            )
            logger.warning(
                f"Creating collection '{self.collection_name}'",
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    'text': VectorParams(
                        size=self.dense_embedder.dimension,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    'text': SparseVectorParams(),
                } if self.sparse_embedder else None,
            )

    async def index(self, chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]

        # Embedder handles batching (20 per call, max 3 concurrent)
        dense_embeddings = await self.dense_embedder.embed(texts)
        sparse_embeddings = await self.sparse_embedder.embed(texts)

        dense_points = await self._collect_dense_points(
            chunks, dense_embeddings,
        )
        sparse_points = await self._collect_sparse_points(
            chunks, sparse_embeddings,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=dense_points + sparse_points,
        )
        logger.info(f"Indexed {len(chunks)} chunks.")

    async def _collect_dense_points(
        self, chunks: list[Chunk],
        dense_embeddings: list[list[float]],
    ) -> list[PointStruct]:
        points = []
        for chunk, dense_embedding in zip(chunks, dense_embeddings):
            point = PointStruct(
                id=int(
                    hashlib.md5(chunk.id.encode()).
                    hexdigest(), 16,
                ) % (10**8),
                vector=dense_embedding,
                payload={
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'pages': chunk.pages,
                    **chunk.metadata,
                },
            )
            points.append(point)

        return points

    async def _collect_sparse_points(
        self, chunks: list[Chunk], sparse_embeddings: list[list[float]],
    ) -> list[PointStruct]:
        points = []
        for chunk, sparse_embedding in zip(chunks, sparse_embeddings):
            point = PointStruct(
                id=int(
                    hashlib.md5(chunk.id.encode()).
                    hexdigest(), 16,
                ) % (10**8),
                sparse_vector=sparse_embedding,
                payload={
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'pages': chunk.pages,
                    **chunk.metadata,
                },
            )
            points.append(point)

        return points
