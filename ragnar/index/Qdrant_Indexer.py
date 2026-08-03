from __future__ import annotations

import hashlib

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import VectorParams

from ragnar.embeddings.ollama_embedding import OllamaEmbedding
from ragnar.index.interfaces.base_indexer import BaseIndexer
from ragnar.models.chunk import Chunk


class QdrantIndexer(BaseIndexer):
    def __init__(
        self, host: str, port: int,
        embedder: OllamaEmbedding, collection_name: str,
        embedding_dim: int = 768,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.embedder = embedder
        self.embedding_dim = embedding_dim
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
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    async def index(self, chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]

        # Embedder handles batching (50 per call, max 3 concurrent)
        embeddings = await self.embedder.embed(texts)

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=int(
                    hashlib.md5(chunk.id.encode()).
                    hexdigest(), 16,
                ) % (10**8),
                vector=embedding,
                payload={
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'pages': chunk.pages,
                    **chunk.metadata,
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Indexed {len(chunks)} chunks.")
