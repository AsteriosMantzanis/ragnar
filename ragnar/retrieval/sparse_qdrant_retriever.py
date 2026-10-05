from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import FieldCondition
from qdrant_client.http.models import Filter
from qdrant_client.http.models import MatchAny

from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
from ragnar.retrieval.interfaces.base_retriever import BaseRetriever
from ragnar.utils.exceptions import CollectionNotFoundError


class SparseQdrantRetriever(BaseRetriever):
    def __init__(
        self,
        url: str = os.getenv('QDRANT_URL', 'http://localhost:6333'),
        embedding: SparseFastEmbedding | None = None,
    ):
        self.client = QdrantClient(url=url)
        # Share the app-wide embedder when given; each new instance loads
        # another copy of the ONNX model into memory.
        self.embedding = embedding or SparseFastEmbedding()

    async def retrieve(
        self, query: str,
        collection_name: str, top_k: int = 20,
    ) -> list[dict]:

        query_sparse_vector = await self.embedding.embed_query(query)

        try:
            search_results = self.client.query_points(
                collection_name=collection_name,
                query=query_sparse_vector,
                using='text-sparse',
                limit=top_k,
                with_payload=True,
            )
        except UnexpectedResponse as e:
            if e.status_code == 404:
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' not found — "
                    'index documents before querying.',
                ) from e
            raise

        return [p.payload for p in search_results.points]

    async def retrieve_hierarchical(
        self, query: str,
        parent_collection_name: str,
        child_collection_name: str,
        linkage_id: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        query_sparse_vector = await self.embedding.embed_query(query)

        try:
            # 1. Search sections
            child_results = self.client.query_points(
                collection_name=child_collection_name,
                query=query_sparse_vector,
                using='text-sparse',
                limit=top_k,
                with_payload=True,
            )

            # 2. Extract parent page IDs
            link_ids = set()
            for result in child_results.points:
                ids = result.payload.get(linkage_id, [])
                link_ids.update(ids)

            # # 3. Fetch parent pages
            parent_results, _ = self.client.scroll(
                collection_name=parent_collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key='chunk_id',
                            match=MatchAny(any=list(link_ids)),
                        ),
                    ],
                ),
                with_payload=True,
            )
        except UnexpectedResponse as e:
            if e.status_code == 404:
                raise CollectionNotFoundError(
                    f"Collection '{parent_collection_name}' or "
                    f"'{child_collection_name}' not found — "
                    'index documents before querying.',
                ) from e
            raise

        # # 4. Return both
        return {
            child_collection_name: [p.payload for p in child_results.points],
            parent_collection_name: [p.payload for p in parent_results],
        }
