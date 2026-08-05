from __future__ import annotations

from qdrant_client import QdrantClient

from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
from ragnar.retrieval.interfaces.base_retriever import BaseRetriever


class SparseQdrantRetriever(BaseRetriever):
    def __init__(self, url: str, collection_name: str):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedding = SparseFastEmbedding()

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_sparse_vector = await self.embedding.embed_query(query)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_sparse_vector,
            using='text-sparse',
            limit=top_k,
            with_payload=True,
        )

        return search_results
