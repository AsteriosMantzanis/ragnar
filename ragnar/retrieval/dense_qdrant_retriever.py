from __future__ import annotations

from qdrant_client import QdrantClient

from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
from ragnar.retrieval.interfaces.base_retriever import BaseRetriever


class DenseQdrantRetriever(BaseRetriever):
    def __init__(self, url: str, collection_name: str):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedding = DenseFastEmbedding()

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_dense_vector = await self.embedding.embed_query(query)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query_vector=query_dense_vector,
            using='text-dense',
            limit=top_k,
            with_payload=True,
        )

        return search_results
