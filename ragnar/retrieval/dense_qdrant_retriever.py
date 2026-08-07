from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition
from qdrant_client.http.models import Filter
from qdrant_client.http.models import MatchAny

from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
from ragnar.retrieval.interfaces.base_retriever import BaseRetriever


class DenseQdrantRetriever(BaseRetriever):
    def __init__(
        self,
        url: str = os.getenv('QDRANT_URL', 'http://localhost:6333'),
    ):
        self.client = QdrantClient(url=url)
        self.embedding = DenseFastEmbedding()

    async def retrieve(
        self, query: str,
        collection_name: str,
        top_k: int = 5,
    ) -> list[dict]:

        query_dense_vector = await self.embedding.embed_query(query)

        search_results = self.client.query_points(
            collection_name=collection_name,
            query=query_dense_vector,
            using='text-dense',
            limit=top_k,
            with_payload=True,
        )

        return [p.payload for p in search_results.points]

    async def retrieve_hierarchical(
        self, query: str,
        parent_collection_name: str, child_collection_name: str,
        top_k: int = 5,
    ) -> dict[str, Any]:

        query_dense_vector = await self.embedding.embed_query(query)

        # 1. Search sections
        section_results = self.client.query_points(
            collection_name=child_collection_name,
            query=query_dense_vector,
            using='text-dense',
            limit=top_k,
            with_payload=True,
        )

        # 2. Extract parent page IDs
        parent_page_ids = set()
        for result in section_results.points:
            page_ids = result.payload.get('parent_page_ids', [])
            parent_page_ids.update(page_ids)

        # # 3. Fetch parent pages
        parent_pages, _ = self.client.scroll(
            collection_name=parent_collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key='chunk_id',
                        match=MatchAny(any=list(parent_page_ids)),
                    ),
                ],
            ),
            with_payload=True,
        )

        # # 4. Return both
        return {
            'sections': [p.payload for p in section_results.points],
            'pages': [p.payload for p in parent_pages],
        }
