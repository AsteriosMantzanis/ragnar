from __future__ import annotations

import os
import uuid
from datetime import datetime
from datetime import timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import VectorParams
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import MatchValue

from ragnar.cache.interfaces.base_sem_cache import BaseSemanticCache
from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding


class QdrantSemanticCache(BaseSemanticCache):

    def __init__(
        self,
        url: str = os.getenv(
            'QDRANT_URL',
            'http://localhost:6333',
        ),
        collection_name: str = 'ragnar_semantic_cache',
        threshold: float = 0.90,
        ttl_seconds: int = 3600,
        embedding: DenseFastEmbedding | None = None,
    ):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.embedding = embedding or DenseFastEmbedding()

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections()

        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding.dimension,
                    distance=Distance.COSINE,
                ),
            )

    async def get(
        self,
        query: str,
        strategy: str,
    ) -> dict[str, Any] | None:

        query_vector = await self.embedding.embed_query(query)

        filter_condition = Filter(
            must=[
                FieldCondition(
                    key='strategy',
                    match=MatchValue(value=strategy),
                ),
            ],
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=filter_condition,
            limit=1,
            with_payload=True,
        )

        if not results.points:
            return None

        match = results.points[0]

        if match.score < self.threshold:
            return None

        payload = match.payload or {}

        # TTL
        created_at = payload.get('created_at')

        if created_at is None:
            return None

        age = (
            datetime.now(timezone.utc).timestamp()
            - created_at
        )

        if age > self.ttl_seconds:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[match.id],
            )
            return None

        return {
            'answer': payload.get('answer', ''),
            'grounding': payload.get('grounding', []),
            'grounded_percentage': payload.get(
                'grounded_percentage',
                0,
            ),
            'sources': payload.get('sources', []),
            'cached': True,
            'cache_score': match.score,
        }

    async def set(
        self,
        query: str,
        strategy: str,
        response: dict[str, Any],
    ) -> None:

        query_vector = await self.embedding.embed_query(query)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={
                'query': query,
                'strategy': strategy,
                'answer': response.get('answer', ''),
                'grounding': response.get('grounding', []),
                'grounded_percentage': response.get(
                    'grounded_percentage',
                    0,
                ),
                'sources': response.get('sources', []),
                'created_at': datetime.now(timezone.utc).timestamp(),
            },
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    async def clear(self) -> None:
        self.client.delete_collection(
            collection_name=self.collection_name,
        )

        self._ensure_collection()
